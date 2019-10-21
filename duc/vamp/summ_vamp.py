
# coding: utf-8

# In[ ]:


import sys
sys.path.append('../')

import os
import time
import datetime
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.manifold import TSNE
from multiprocessing import cpu_count

from duc import DUC
from vamp.vamp_model import vamp
from utils import linear_anneal, log_Normal_diag, log_Normal_standard
from utils import load_pickle


# In[ ]:


# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

max_len = 64
batch_size = 32
#pseudo_size = 100
#splits = ['train', 'valid', 'test']

# Penn TreeBank (PTB) dataset
data_path = '../data'
#datasets = {split: DUC(root=data_path, split=split) for split in splits}
datasets = DUC(root=data_path, split='summ')

pseudo_dataset = DUC(root=data_path, split='test')
#datasets['valid'] = datasets['valid'][pseudo_size:]



# In[ ]:


len(pseudo_dataset)


# In[ ]:


# dataloader
dataloaders = DataLoader(datasets,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=cpu_count(),
                                 pin_memory=torch.cuda.is_available()
                        )
                                 
symbols = datasets.symbols

pseudo_dataloader = DataLoader(pseudo_dataset,
                                batch_size=1390,
                                pin_memory=torch.cuda.is_available())


# In[ ]:


# vamp model
embedding_size = 300
hidden_size = 256
latent_dim = 32
dropout_rate = 0.5
model = vamp(vocab_size=datasets.vocab_size,
               embed_size=embedding_size,
               time_step=max_len,
               hidden_size=hidden_size,
               z_dim=latent_dim,
               dropout_rate=dropout_rate,
               bos_idx=symbols['<bos>'],
               eos_idx=symbols['<eos>'],
               pad_idx=symbols['<pad>'])
model = model.to(device)


# In[ ]:


# pseudo input
pseudo_inputs, _, _, pseudo_lengths, _ = next(iter(pseudo_dataloader))
pseudo_inputs = pseudo_inputs.to(device)
pseudo_lengths = pseudo_lengths.to(device)

pseudo_sorted_len, pseudo_sorted_idx = torch.sort(pseudo_lengths, descending=True)
pseudo_inputs = pseudo_inputs[pseudo_sorted_idx]


# In[ ]:


# objective function
learning_rate = 0.001
criterion = nn.NLLLoss(size_average=False, ignore_index=symbols['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# negative log likelihood
def NLL(logp, target, length):
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(-1))
    return criterion(logp, target)


# In[ ]:


def log_prior(z_q):
    z_p_mu, z_p_logvar = model.encoder(pseudo_inputs, pseudo_sorted_len)
    z_q_expand = z_q.unsqueeze(1)
    means = z_p_mu.unsqueeze(0)
    logvars = z_p_logvar.unsqueeze(0)

    a = log_Normal_diag(z_q_expand, means, logvars, dim=2) - math.log(1390)#pseudo_size)
    a_max, _ = torch.max(a, 1)

    log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))

    
    return log_prior


# In[ ]:


# training setting
epoch = 20
print_every = 50

# training interface
step = 0
tracker = {'ELBO': [], 'NLL': [], 'KL': [], 'KL_weight': []}
start_time = time.time()
for ep in range(epoch):
    # learning rate decay
    if ep >= 10 and ep % 2 == 0:
        learning_rate = learning_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    
    
    model.train() 
    totals = {'ELBO': 0., 'NLL': 0., 'KL': 0., 'words': 0}

    for itr, (enc_inputs, dec_inputs, targets, lengths, labels,_) in enumerate(dataloaders):
        bsize = enc_inputs.size(0)
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # forward
        logp, z_q, mu, logvar, _ = model(enc_inputs, dec_inputs, lengths, labels)

        # calculate loss
        NLL_loss = NLL(logp, targets, lengths + 1)
        # KL loss
        log_p_z = log_prior(z_q)
        log_q_z = log_Normal_diag(z_q, mu, logvar, dim=1)
        KL_loss = torch.sum(-(log_p_z - log_q_z))
        KL_weight = linear_anneal(step, len(dataloaders) * 10)
        loss = (NLL_loss + KL_weight * KL_loss) / bsize

        # cumulate
        totals['ELBO'] += loss.item() * bsize
        totals['NLL'] += NLL_loss.item()
        totals['KL'] += KL_loss.item()
        totals['words'] += torch.sum(lengths).item()

        # backward and optimize
        
        step += 1
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # track
        tracker['ELBO'].append(loss.item())
        tracker['NLL'].append(NLL_loss.item() / bsize)
        tracker['KL'].append(KL_loss.item() / bsize)
        tracker['KL_weight'].append(KL_weight)


        if(False):
            # print statistics
            if itr % print_every == 0 or itr + 1 == len(dataloaders):
                print("summ Batch %04d/%04d, ELBO-Loss %.4f, "
                      "NLL-Loss %.4f, KL-Loss %.4f, KL-Weight %.4f"
                      % ( itr, len(dataloaders),
                         tracker['ELBO'][-1], tracker['NLL'][-1],
                         tracker['KL'][-1], tracker['KL_weight'][-1]))

    samples = len(datasets)
    #print('\n')
    print("summ Epoch %02d/%02d, ELBO %.4f, NLL %.4f, KL %.4f, PPL %.4f"
          % ( ep, epoch, totals['ELBO'] / samples,
             totals['NLL'] / samples, totals['KL'] / samples,
             math.exp(totals['NLL'] / totals['words'])))
    
    #print('\n')
    # save checkpoint
    #checkpoint_path = os.path.join(save_path, "E%02d.pkl" % ep)
    #torch.save(model.state_dict(), checkpoint_path)
    #print("Model saved at %s\n" % checkpoint_path)
end_time = time.time()
print('Total cost time',
      time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))


# In[ ]:



model_cpu = model.cpu()
torch.no_grad()


# In[ ]:


n_sent = []
for i in range(1,46):
    k = 0
    for j in range(len(datasets)):
        if datasets[j][4] == i:
            k+=1
    n_sent.append(k)

all_sent = sum(n_sent)


# In[ ]:


# cpu
a = 0
b = 0
id_all = []
for i in range(45):
    b += n_sent[i]
    print(a,b)
    data = datasets[a:b]
    a += n_sent[i]
    loader = DataLoader(data, batch_size=n_sent[i])
    enc_inputs, dec_inputs, targets, lengths, labels, sent_id = next(iter(loader))
    _, sorted_idx = torch.sort(lengths, descending=True)
    sent_id = np.array(sent_id)
    sent_id = sent_id[sorted_idx]
    
    logp, z, mu, logvar, labels = model_cpu(enc_inputs, dec_inputs, lengths, labels)
    
    z_mean = torch.sum(mu,0)
    inner = []
    for latent in mu:
        inner.append(torch.abs(torch.dot(z_mean, latent) / torch.norm(z_mean) / torch.norm(latent)).item())
    inner = np.array(inner)
    inner_sort = np.argsort(-inner)
    sent_id = sent_id[inner_sort]
    id_all.append(sent_id)


# In[ ]:


path = os.path.join('../data','summ')
topics = os.listdir(path)
topics.sort()
i = 0
summ = []
for topic in topics:
    i += 1
    select = []
    with open(os.path.join(path,topic),encoding='UTF-8') as f:
        lines = f.readlines()
        for j in range(30):
            identity = id_all[i-1][j]
            select.append(lines[identity-1])
    summ.append(select)


# In[ ]:


path = os.path.join('../data','models')
topics = os.listdir(path)
topics.sort()

g_truth = []
for i in range(45):
    ground = []
    for j in range(4):
        with open(os.path.join(path,topics[i*4+j]),encoding='ISO-8859-1') as f:
            ground.append(f.read())
    truth = ' '.join(ground)
    g_truth.append(truth)


# In[ ]:


from rouge import Rouge
rouge = Rouge()

scores = []

for i in range(45):
    hypo = ' '.join(summ[i])
    score = []
    
    ref = g_truth[i]
    scores.append(rouge.get_scores(hypo, ref, avg=True))
    #scores.append(score)


# In[ ]:


r1f = 0
r1p = 0
r1r = 0
r2f = 0
r2p = 0
r2r = 0
rlongf = 0
rlongp = 0
rlongr = 0
for i in range(45):
    r1f += scores[i]['rouge-1']['f'] * n_sent[i] / all_sent
    r1p += scores[i]['rouge-1']['p'] * n_sent[i] / all_sent
    r1r += scores[i]['rouge-1']['r'] * n_sent[i] / all_sent
    r2f += scores[i]['rouge-2']['f'] * n_sent[i] / all_sent
    r2p += scores[i]['rouge-2']['p'] * n_sent[i] / all_sent
    r2r += scores[i]['rouge-2']['r'] * n_sent[i] / all_sent
    rlongf += scores[i]['rouge-l']['f'] * n_sent[i] / all_sent
    rlongp += scores[i]['rouge-l']['p'] * n_sent[i] / all_sent
    rlongr += scores[i]['rouge-l']['r'] * n_sent[i] / all_sent
print(r1f,r1p,r1r)
print(r2f,r2p,r2r)
print(rlongf,rlongp,rlongr)


# In[ ]:


print('%.3f %.3f %.3f' % (r1f,r1p,r1r))
print('%.3f %.3f %.3f' % (r2f,r2p,r2r))
print('%.3f %.3f %.3f' % (rlongf,rlongp,rlongr))


# In[ ]:


# calculate au

model = model.to(device)

delta = 0.01
with torch.no_grad():
    model.eval()
    cnt = 0
    for itr, (enc_inputs, dec_inputs, targets, lengths, labels,_) in enumerate(dataloaders):
        bsize = enc_inputs.size(0)
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # forward
        logp, z, mu, logvar, _ = model(enc_inputs, dec_inputs, lengths, labels)
        
        if cnt == 0:
            mu_sum = mu.sum(dim=0, keepdim=True)
        else:
            mu_sum = mu_sum + mu.sum(dim=0, keepdim=True)
        cnt += mu.size(0)
        
    mu_mean = mu_sum / cnt
        
    cnt = 0
    for itr, (enc_inputs, dec_inputs, targets, lengths, labels,_) in enumerate(dataloaders):
        bsize = enc_inputs.size(0)
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # forward
        logp, z, mu, logvar, _ = model(enc_inputs, dec_inputs, lengths, labels)
        
        if cnt == 0:
            var_sum = ((mu - mu_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mu - mu_mean) ** 2).sum(dim=0)
        cnt += mu.size(0)
        
    au_var = var_sum / (cnt - 1)
    
    print((au_var >= delta).sum().item())
    print(au_var)


# In[ ]:


# plot KL curve
fig, ax1 = plt.subplots()
lns1 = ax1.plot(tracker['KL_weight'], 'b', label='KL term weight')
ax1.set_ylim([-0.05, 1.05])
ax1.set_xlabel('Step')
ax1.set_ylabel('KL term weight')
ax2 = ax1.twinx()
lns2 = ax2.plot(tracker['KL'], 'r', label='KL term value')
ax2.set_ylabel('KL term value')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102),
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


# In[ ]:


# latent space visualization
if(True):
    features = np.empty([len(datasets), latent_dim])
    feat_label = np.empty(len(datasets))
    for itr, (enc_inputs, dec_inputs, _, lengths, labels,_) in enumerate(dataloaders):
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        _, _, mu, _, labels = model(enc_inputs, dec_inputs, lengths, labels)
        start, end = batch_size * itr, batch_size * (itr + 1)
        features[start:end] = mu.data.cpu().numpy()
        feat_label[start:end] = labels.data.cpu().numpy()

    
    


# In[ ]:


tsne_z = TSNE(n_components=2,perplexity=10).fit_transform(features)
tracker['z'] = tsne_z
tracker['label'] = feat_label


# In[ ]:


plt.figure()
for i in range(2,8):
    plt.scatter(tsne_z[np.where(feat_label == i), 0], tsne_z[np.where(feat_label == i), 1], s=10, alpha=0.5)
plt.show()


# In[ ]:


# save learning results
sio.savemat("vamp_summ.mat", tracker)


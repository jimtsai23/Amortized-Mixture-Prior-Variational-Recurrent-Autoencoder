import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, z_dim, pad_idx):
        super(LSTMEncoder, self).__init__()
        # input
        
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size * 2, z_dim * 2)

    def forward(self, input_seq, length):
        # embed input
        embedded_input = self.embedding(input_seq)
        #embedded_input = embedded_input + torch.randn_like(embedded_input).normal_(0,self.sigma)

        # RNN forward
        pack_input = pack_padded_sequence(embedded_input, length,
                                          batch_first=True)
        _, (h, c) = self.rnn(pack_input)

        # produce mu and logvar
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.output(hidden), 2, dim=-1)

        return mu, logvar    
    
class vamp(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(vamp, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.encoder = LSTMEncoder(vocab_size, embed_size,
                                   hidden_size, z_dim, pad_idx)
        # decoder
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        self.init_h = nn.Linear(z_dim, hidden_size)
        self.init_c = nn.Linear(z_dim, hidden_size)
        self.skip = nn.Linear(z_dim, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, enc_input, dec_input, length, labels):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]
        labels = labels[sorted_idx]

        # encode
        mu, logvar = self.encoder(enc_input, sorted_len)
        z = self.reparameterize(mu, logvar)

        # decode
        embedded_input = self.embedding(dec_input)
        #res = self.skip(z).unsqueeze(1).expand(-1,self.time_step+1,-1)
        #embedded_input = embedded_input + res
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        h_0, c_0 = self.init_h(z), self.init_c(z)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        pack_output, _ = self.rnn(pack_input, hidden)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, z, mu, logvar, labels

    def inference(self, z):
        # set device
        tensor = torch.LongTensor
        if torch.cuda.is_available():
            tensor = torch.cuda.LongTensor

        # initialize hidden state
        batch_size = z.size(0)
        h_0, c_0 = self.init_h(z), self.init_c(z)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))

        # RNN forward
        symbol = tensor(batch_size, self.time_step + 1).fill_(self.pad_idx)
        for t in range(self.time_step + 1):
            if t == 0:
                input_seq = tensor(batch_size, 1).fill_(self.bos_idx)
            embedded_input = self.embedding(input_seq)
            output, hidden = self.rnn(embedded_input, hidden)
            logit = self.output(output)
            _, sample = torch.topk(logit, 1, dim=-1)
            input_seq = sample.squeeze(-1)
            symbol[:, t] = input_seq.squeeze(-1)

        return symbol


import os
import numpy as np
from collections import Counter

from utils import save_pickle
from nltk.corpus import stopwords


root = 'data'
max_len = 64
vocab_size = 12000
stop_words = set(stopwords.words('english')) 

print("Building vocabulary from Yelp data")
counter = Counter()
with open(os.path.join(root, 'train.txt')) as f:
    for line in f:
        words = line.strip().split()[1:max_len]
        counter.update(words)
        
pre_vocab = [(word, freq) for word, freq in counter.most_common()]# if freq > 5]
#pre_vocab = [(word, freq) for word, freq in pre_vocab if freq < 15500]
#print(pre_vocab)

if True:
    word_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

    vocab = [word for word, freq in pre_vocab]
    #vocab = [w for w in vocab if not w in stop_words]
    for word in vocab[:vocab_size - 2]:
        word_to_idx[word] = len(word_to_idx)

        
    # exclude <bos> and <pad> symbols
    print("Vocabulary size: %d" % (len(word_to_idx) - 2))


    save_pickle(word_to_idx, os.path.join(root, 'vocab.pkl'))



    splits = ['train', 'valid', 'test']
    num_sents, num_words = 0, 0
    func = lambda seq: np.array([
        word_to_idx.get(symbol, word_to_idx['<unk>']) for symbol in seq])
    for split in splits:
        print("Creating %s Yelp data" % split)
        data = []
        with open(os.path.join(root, "%s.txt" % split)) as f:
            for line in f:
                words = line.strip().split()[:max_len + 1]
                label, raw_words = int(words[0]), words[1:]
                words = [w for w in raw_words] # if not w in stop_words]
                length = len(words)
                paddings = ['<pad>'] * (max_len - length)
                enc_input = func(words + paddings)
                dec_input = func(['<bos>'] + words + paddings)
                target = func(words + ['<eos>'] + paddings)
                data.append((enc_input, dec_input, target, length, label))
                num_words += length
        print("%s samples: %d" %(split.capitalize(), len(data)))
        save_pickle(data, os.path.join(root, "%s.pkl" % split))
        num_sents += len(data)

    print("Average length: %.2f" %(num_words / num_sents))

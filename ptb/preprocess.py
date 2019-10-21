import os
import numpy as np
from collections import Counter

from utils import save_pickle


root = 'data2'
max_len = 64
vocab_size = 8000

print("Building vocabulary from PTB data")
counter = Counter()
with open(os.path.join(root, 'ptb.train.txt')) as f:
    for line in f:
        words = line.strip().split()[:max_len]
        counter.update(words)
                
word_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

vocab = [word for word, _ in counter.most_common()]
for word in vocab[:vocab_size - 2]:
    word_to_idx[word] = len(word_to_idx)

print(counter.most_common()[:vocab_size - 2])    
if(False):    
    def idx_to_word(word_to_idx):
        return {val: key for key, val in word_to_idx.items()}
    print(idx_to_word(word_to_idx))
            

# exclude <bos> and <pad> symbols
print("Vocabulary size: %d" % (len(word_to_idx) - 2))
#save_pickle(word_to_idx, os.path.join(root, 'ptb.vocab.pkl'))

splits = ['train', 'valid', 'test']
num_sents, num_words = 0, 0
func = lambda seq: np.array([
    word_to_idx.get(symbol, word_to_idx['<unk>']) for symbol in seq])
for split in splits:
    print("Creating %s PTB data" % split)
    data = []
    with open(os.path.join(root, "ptb.%s.txt" % split)) as f:
        for line in f:
            words = line.strip().split()[:max_len]
            length = len(words)
            paddings = ['<pad>'] * (max_len - length)
            enc_input = func(words + paddings)
            dec_input = func(['<bos>'] + words + paddings)
            target = func(words + ['<eos>'] + paddings)
            data.append((enc_input, dec_input, target, length))
            num_words += length
    print("%s samples: %d" %(split.capitalize(), len(data)))
    #save_pickle(data, os.path.join(root, "ptb.%s.pkl" % split))
    num_sents += len(data)

print("Average length: %.2f" %(num_words / num_sents))

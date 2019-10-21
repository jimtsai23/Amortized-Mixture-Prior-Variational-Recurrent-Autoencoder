import os
import re
import random
import numpy as np
from collections import Counter
from nltk import word_tokenize

from utils import save_pickle


root = 'data'
ratios = [('train', 0.85), ('valid', 0.05), ('test', 0.1)]
max_len = 80
vocab_size = 20000

data = []
for split in ['train', 'test']:
    #folders = ['pos', 'neg', 'unsup'] if split == 'train' else ['pos', 'neg']
    #for folder in folders:
    for folder, label in [('pos', 1), ('neg', 0)]: ###
        folder_path = os.path.join(root, split, folder)
        files = os.listdir(folder_path)
        for file in files:
            with open(os.path.join(folder_path, file)) as f:
                text = f.read()
                text = re.sub(r'<[^>]+>', '', text) # remove HTML tags
                text = re.sub(r'\(.*?\)', '', text) # remove bracket contents
                # text = re.sub(r'', '', text) # removie unknown symbol

                # replace punctuation characters with spaces
                filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
                translate_dict = dict((c, " ") for c in filters)
                translate_map = str.maketrans(translate_dict)
                text = text.translate(translate_map)

                tokens = word_tokenize(text)
                line = [str(label)] + tokens ###
                data.append(' '.join(line)) ###
random.shuffle(data)

num_samples = len(data)
for split, ratio in ratios:
    with open(os.path.join(root, "%s.txt"%split), 'w') as f:
        length = int(num_samples * ratio)
        f.write('\n'.join(data[:length]))
    data = data[length:]

print("Building vocabulary from IMDB data")
counter = Counter()
with open(os.path.join(root, 'train.txt')) as f:
    for line in f:
        words = line.strip().lower().split()[:max_len]
        counter.update(words)

word_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
vocab = [word for word, _ in counter.most_common()]
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
    print("Creating %s IMDB data" % split)
    data = []
    with open(os.path.join(root, "%s.txt" % split)) as f:
        for line in f:
            words = line.strip().lower().split()[:max_len + 1] ###
            label, words = int(words[0]), words[1:] ###
            length = len(words)
            paddings = ['<pad>'] * (max_len - length)
            enc_input = func(words + paddings)
            dec_input = func(['<bos>'] + words + paddings)
            target = func(words + ['<eos>'] + paddings)
            data.append((enc_input, dec_input, target, length, label)) ###
            num_words += length
    print("%s samples: %d" %(split.capitalize(), len(data)))
    save_pickle(data, os.path.join(root, "%s.pkl" % split))
    num_sents += len(data)

print("Average length: %.2f" %(num_words / num_sents))

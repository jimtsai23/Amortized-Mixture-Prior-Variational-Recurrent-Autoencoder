import requests
import os
import numpy as np
from bs4 import BeautifulSoup
from nltk import word_tokenize
from collections import Counter
import random

from utils import save_pickle


root = 'data'
ratios = [('train', 0.85), ('valid', 0.05), ('test', 0.1)]
max_len = 64
vocab_size = 16000


data = []
path = os.path.join(root,'main')
topics = os.listdir(path)
i = 0
for topic in topics:
    i += 1
    arts = os.listdir(os.path.join(path,topic))
    j = 0
    for art in arts:
        j += 1
        with open(os.path.join(path,topic,art),encoding='UTF-8') as f:
            #lines = unicode(f.read(), errors='ignore')
            lines = f.read()
            #print(type(lines))
            #print(i,j)
            soup = BeautifulSoup(lines, 'html.parser')
            for text in soup.find_all('p'):
                # replace punctuation characters with spaces
                text = text.get_text()
                filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
                translate_dict = dict((c, " ") for c in filters)
                translate_map = str.maketrans(translate_dict)
                text = text.translate(translate_map)
                tokens = word_tokenize(text)
                lines = [str(i)] + [str(j)] + tokens
                if(len(tokens)==0):
                    break
                else:
                    data.append(' '.join(lines))

if True:
    random.shuffle(data)

    num_samples = len(data)
    for split, ratio in ratios:
        with open(os.path.join(root, "%s.txt"%split), 'w') as f:
            length = int(num_samples * ratio)
            f.write('\n'.join(data[:length]))
        data = data[length:]

    print("Building vocabulary from DUC data")
    counter = Counter()
    with open(os.path.join(root, 'train.txt')) as f:
        for line in f:
            words = line.strip().lower().split()[:max_len]
            counter.update(words)

    word_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    vocab = [word for word, freq in counter.most_common() if freq > 5]
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
        print("Creating %s DUC data" % split)
        data = []
        with open(os.path.join(root, "%s.txt" % split)) as f:
            for line in f:
                words = line.strip().lower().split()[:max_len + 2]
                topic, art, words = int(words[0]), int(words[1]), words[2:] ###
                length = len(words)
                paddings = ['<pad>'] * (max_len - length)
                enc_input = func(words + paddings)
                dec_input = func(['<bos>'] + words + paddings)
                target = func(words + ['<eos>'] + paddings)
                data.append((enc_input, dec_input, target, length, topic)) ###
                num_words += length
        print("%s samples: %d" %(split.capitalize(), len(data)))
        save_pickle(data, os.path.join(root, "%s.pkl" % split))
        num_sents += len(data)

    print("Average length: %.2f" %(num_words / num_sents))
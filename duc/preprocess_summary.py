import requests
import os
import numpy as np
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from collections import Counter
import random

from utils import save_pickle


root = 'data'
#ratios = [('train', 0.85), ('valid', 0.05), ('test', 0.1)]
max_len = 64
vocab_size = 16000


data = []
path = os.path.join(root,'main')
topics = os.listdir(path)
topics.sort()
i = 0
if True:
    for topic in topics:
        i += 1
        arts = os.listdir(os.path.join(path,topic))
        arts.sort()
        with open(os.path.join(root, 'summ', "%.2d.txt" % i), 'w') as file:
            for art in arts:
                with open(os.path.join(path,topic,art),encoding='ISO-8859-1') as f:
                    essay = f.read()
                    #print(type(essay))
                    soup = BeautifulSoup(essay, 'html.parser')

                    for text in soup.find_all('p'):
                        text = text.get_text()
                        filters = '!"\#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
                        translate_dict = dict((c, "") for c in filters)
                        translate_map = str.maketrans(translate_dict)
                        text = text.translate(translate_map)
                        sent = sent_tokenize(text)
                        for s in sent:
                            file.write(s+'\n')
                            
            

path = os.path.join(root,'summ')
topics = os.listdir(path)
topics.sort()
i = 0

if True:
    for topic in topics:
        i += 1
        with open(os.path.join(path,topic),encoding='UTF-8') as f:
            lines = f.readlines()
            #print(len(lines))
            j = 0
            for line in lines:
                j += 1
                #filters = '!"\#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
                #filters = '"\t\n'
                #translate_dict = dict((c, " ") for c in filters)
                #translate_map = str.maketrans(translate_dict)
                #line = line.translate(translate_map)
                tokens = word_tokenize(line)
                sent = [str(i)] + [str(j)] + tokens
                if(len(tokens)==0):
                    break
                else:
                    data.append(' '.join(sent))
        
                    

    with open(os.path.join(root, "summ.txt"), 'w') as f:
        f.write('\n'.join(data))

if True:
    num_samples = len(data)
    print(num_samples)

    print("Building vocabulary from DUC data")
    counter = Counter()
    with open(os.path.join(root, 'summ.txt')) as f:
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

    num_sents, num_words = 0, 0
    func = lambda seq: np.array([
        word_to_idx.get(symbol, word_to_idx['<unk>']) for symbol in seq])

    print("Creating summ DUC data")
    data = []
    with open(os.path.join(root, "summ.txt" )) as f:
        for line in f:
            words = line.strip().lower().split()[:max_len + 2]
            topic, sent_id, words = int(words[0]), int(words[1]), words[2:] ###
            length = len(words)
            paddings = ['<pad>'] * (max_len - length)
            enc_input = func(words + paddings)
            dec_input = func(['<bos>'] + words + paddings)
            target = func(words + ['<eos>'] + paddings)
            data.append((enc_input, dec_input, target, length, topic, sent_id)) ###
            num_words += length
    print("SUMM samples: %d" %(len(data)))
    save_pickle(data, os.path.join(root, "summ.pkl"))
    num_sents += len(data)

    print("Average length: %.2f" %(num_words / num_sents))



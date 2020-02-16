import string
import re
import os
import io
import sys
import csv
import six
import itertools
from collections import Counter
from collections import defaultdict, OrderedDict

import torch
from torchtext.vocab import Vectors, Vocab

from dataAugment.dataAugment import *

# 前処理
def preprocessing_text(text):
    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ",") or (p == ":") or (p == "<" )or (p == ">"):
            continue
        else:
            text = text.replace(p, " ")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = re.sub(r'[0-9 ０-９]', '0', text)

    return text

# 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）
def tokenizer_punctuation(text):
    return text.strip().split(':')

# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


def unicode_csv_reader(unicode_csv_data, **kwargs):
    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    if six.PY2:
        # csv.py doesn't do Unicode; encode temporarily as UTF-8:
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), **kwargs)
        for row in csv_reader:
            # decode UTF-8 back to Unicode, cell by cell:
            yield [cell.decode('utf-8') for cell in row]
    else:
        for line in csv.reader(unicode_csv_data, **kwargs):

            yield line


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_len=64, vectors=None, 
                 min_freq=10, specials=[], phase='val'):
        self.keys = ['Date', 'Code', 'State', 'Next_State', 'Reward']
        self.string_keys = ['State', 'Next_State']
        self.tensor_keys = ['Reward'] + self.string_keys
        self.data_list = {k: [] for k in self.keys}
        
        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            reader = unicode_csv_reader(f, delimiter='\t')
            for line in reader:
                for k, x in zip(self.keys, line):
                    if k in self.string_keys:
                        self.data_list[k].append(x.split(':'))
                    elif k in self.tensor_keys:
                        self.data_list[k].append(float(x))
                    else:
                        self.data_list[k].append(x)
        
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.init_token = '<cls>'
        self.eos_token = '<eos>'
        self.max_len = max_len
        self.fix_len = self.max_len + (self.init_token, self.eos_token).count(None) - 2
        self.specials = specials

        self.words = list(itertools.chain.from_iterable(self.data_list['State']))
        self.counter = Counter(self.words)

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + self.specials
            if tok is not None))
        self.vocab = Vocab(self.counter,
                           specials=specials, 
                           vectors=vectors, 
                           min_freq=min_freq) 
        self.padded_list = self.pad(self.data_list)
        self.tensor_list = self.numericalize(self.padded_list)

        stopwords = []
        for w in ['<cls>', '<eos>', '<pad>', '<span>']:
            stopwords.append(self.vocab.stoi[w])

        self.transform = DataTransform(self.vocab, stopwords)
        self.phase = phase

    def pad(self, data):
        padded = {k: [] for k in self.keys}
        for key, val in data.items():
            if key in self.string_keys:
                arr = []
                for x in val:
                    arr.append(
                        ([self.init_token])
                        + list(x[:self.fix_len])
                        + ([self.eos_token])
                        + [self.pad_token] * max(0, self.fix_len - len(x)))
                padded[key] = arr
            else:                
                padded[key] = val

        return padded

    def numericalize(self, padded):
        tensor = {k: [] for k in self.keys}
        for key, val in padded.items():
            if key in self.string_keys:
                arr = []
                for ex in val:
                    arr.append([self.vocab.stoi[x] for x in ex])
                tensor[key] = torch.LongTensor(arr).to('cpu')
            elif key in self.tensor_keys:
                tensor[key] = torch.FloatTensor(val).to('cpu')
            else:                
                tensor[key] = val

        return tensor

    def __len__(self):
        return len(self.tensor_list['State'])

    def __getitem__(self, i):
        arr = {k: [] for k in self.keys}
        for key in self.keys:
            data = self.tensor_list[key][i]
            if key == 'State':
                data = self.transform(data, self.phase)

            arr[key] = data

        return arr


class DataTransform:
    def __init__(self, vectors, stopwords):
        self.data_transform = {
            'train': Compose([
                RandomSwap(vectors, aug_p=0.1, stopwords=stopwords),
                RandomSubstitute(vectors, aug_p=0.1, stopwords=stopwords),
            ]),
            'val': Compose([
            ])
        }
    
    def __call__(self, text, phase):
        return self.data_transform[phase](text)

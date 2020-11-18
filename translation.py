# Translation with a Sequence to Sequence Network and Attention

# creating a neural network to translate from French to English

# [To-Do] Download the data and extract it to the current directory
# https://download.pytorch.org/tutorial/data.zip

from __future__ import unicode_literals,print_function,division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS", 1: "EOS"}
        self.n_words = 2 # count sos and eos

    def addSentence(self,sentence):
        for word in sentenc.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

# turn a unicode to ascii
def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD',s)
            if unicodedata.category(c) != 'Mn'
            )

# lowercase, trim and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[.!?])", r" \1", r)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ",s)
    return s



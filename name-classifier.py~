# Classifying names with character-level RNN

# We'll be training on a few thousands surnames from 18 languages or origin, and predict which language a name is from based on the spelling.

# [To-Do] download and extract data in the current directory
# https://download.pytorch.org/tutorial/data.zip

# Preparing the data

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import time
import math
import random
import unicodedata
import string
import torch
import torch.nn as nn

def findFiles(path):
    return glob.glob(path)

# print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

# turn a unicode string to plain ascii
def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD',s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
            )

# print(unicodeToAscii('Slusarski')

# build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_catgories)

# print(category_lines['Italian'][:5])

# Turning names into tensors

# find letter index from all_letter
def letterToIndex(letter):
    return all_letters.find(letter)

# turn a letter into a <1 x n_letters> tensor
def letterToTensor(letter):
    tensor = torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line),1, n_letters)
    for li,letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# print(letterToTensor('J'))
# print(lineToTensor('Jones').size())

# Creating the network

class RNN(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden),1)
        hidden = self.i2h(combined)
        output = self.i20(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# Training the Network

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i],category_i

# get a training example ( a name and its language)
def randomChoice(l):
    return l[random.randint(0,len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)],dtype=torch.lang)
    line_tensor = lineToTensor(line)
    return categor, line, category_tensor, line_tensor

for i in range(10):
    category,line,category_tensor,line_tensor = randomTrainingExample()
    print('category = ',category,'/ line = ', line)



criterion = nn.NLLLoss()

learning_rete = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output,hidden = rnn(line_tensor[i],hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # add parameters' gradient to their values
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output,loss.item()

n_iter = 100000
print_every = 5000
plot_every = 1000

# keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m,s)

start = time.time()

for iter in range(1,n_iter + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    curren_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '~' if guess == category else 'X (%s)' % categorry
        print('%d %d%% (%s) %.4f %s %s %s' % (iter,iter/n_iters * 100,timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Plotting the Results
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

# Evaluating the Results









# Classifying names with a character level RNN

import glob
import unicodedata
import string
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Vatiable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Preparing the Data

all_filenames = glob.glob('./data/names/*.txt')
all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) != 'Mn' and c in all_letters)


# build the catergory line dictionary, a list of names per language 

category_line = {}
all_categories = []

# read a fule and split into line
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in all_filenames:
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# 2. Turning names inot tensors

def letter_to_tensor(letter):
    tensor = torch.zeros(1,n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor

# an array of one-hot letter vectors

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor


# 3. Creating the Network

class RNN(nn.Module):
    def __init(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

# 4. Manually testing the network

n_hidden = 28

rnn = RNN(n_letters, n_hidden, n_categories)

input = Variable(letter_to_tensor('A'))
hidden = rnn.init_hidden()

output, next_hidden = rnn(input, hidden)
print("Output Size = ",output.size())

# pre-computing batches of tensors
input = Variable(line_to_tensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden= trnn(input[0], hidden)
print(output)

# 5. Preparing for Training

def category_from_output(output):
    top_n, top_i = output.data.topk(1) # tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

# get a traing example

def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_line[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category,line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_pair()
    print('category = ',category, '/ line =', line)


# 6. Training the Network

# loss function
criterion = nn.NLLLoss()

learning_rate = 0.005
optimizer = torch.optimi.SGD(rnn.parameters(), lr=learning_rate)

def train(category_tensor, line_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output,hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]


n_epochs = 10000
print_every = 100
plot_every = 500

# keep track of losses for plotting
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floors(s / 60)
    s -= m * 60
    return '%dm %ds' % (m,s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    # get a random training input and targe
    category_lin, line, category_tensor, line_tensor = random_training_pair()

    # print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '~' if guess == category else 'X (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

        if epoch % plo_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

# Plotting the results

plt.figure()
plt.plot(all_losses)

# Evaluating the Results

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# return an output given a line
def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

# go through a bunch of example and record which are correctly guessed

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_pair()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipltLocator(1))

plt.show()

# Running on user input

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    output = evaluate(Variable(line_to_tensor(input_line)))

    # get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_idnex == topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

predict('Ujjawal')
predict('Jackson')
predict('Hideo')



















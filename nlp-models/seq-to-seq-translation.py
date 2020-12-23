"""
Translation with a Sequence to Sequence Network and Attention

We'll use PyTorch to translate from French to English.

Data:
    Download and ectract from http://www.manythings.org/anki/
"""

import random
import re
import math
import unicodedata
import time
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

USE_CUDA = True
# Indexing Words

sos_token = 0
eos_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "sos", 1: "eos"}
        self.n_words = 2

    def index_words(self, sentence):
        for word in sentence.split(' '):
            return index_word(word)
    def index_word(self, sentence):
        if word not in self.word2index:
            self.word2index[word] = n_words
            self.word2count[word] = 1
            self.index2word[n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Reading and Decoding Files

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicode.category(c) != 'Md'
    )

# lowercase, trim and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().split())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# to read the data files we'll split the file into line and then lines into pairs

def read_lang(lang1, lang2, reverse=False):
    print('Reading lines...')

    # read the file and split into lines
    lines = open('./data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')

    # split every lines into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # reverse pairs, make lang instances
    if reverse:
        pairs = [List(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    
    return input_lang, output_lang, pairs

# trim the data to only relatively short and simple sentences.

max_length = 10

good_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re "
)

def filter_pair(p):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length and \
            p[1].stratswith(good_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pairs)]


def prepare_data(lang1_name, lang2_name, pairs):
    input_lang, output_lang, pairs = read_lang(lang1_name, lang2_name, reverse)
    print('Read %s sentences.' % len(pairs))

    pairs = filter_pairs(pairs)
    print('Trimmed to %s sentence pairs.' % len(pairs))

    print('Indexing words...')
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

print(random.choice(pairs))

# Turning training data inot Tensors/Variables

# return a list of indexes, one for each word in the sentence

def index_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = index_from_sentence(lang, sentence)
    indexes.append(eos_token)
    var = Variable(torch.LongTensor(indexes).view(-1,1))
    if USE_CUDA: var = var.cuda()
    return var
def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)

# Building the Models

# The Encoder

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embediing = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
    
    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1,-1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

# Bahdanau Attention Decoder

class BahdAttDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.outpu_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(droput_p)
        self.attn = GeneralAttn(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        word_embedded = self.embedding(word_input).view(1,1,-1)
        word_embedded = self.dropout(word_embedded)

        # calculate attention weights and apply to encoder outputs
        attn_weight = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        # combine embedded input word and attened context, run through RNN
        rnn_input = torch.cat((word_embedded, context),2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # final output layer
        output = output.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output,context),1)))
        return output,hidden, attn_weights

class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=max_length):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size *2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1,hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # create variable to store attention energies
        attn_categories = Variable(torch.zeros(seq_len))
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output),1))
            energy = self.other.dot(energy)
            return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # get the embedding of the current input word 
        word_embedded = self.embedding(word_input).view(1,1,-1)

        # combine embedded input word and last context, run through outputs
        rnn_input = torch.cat((word_embedded, last_contex.unsqueeze(0)),2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # calculate attention from current RNN state and all encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_ouputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        # final output layer using the RNN hidden state
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat(rnn_output, context),1)))

        return output, context, hidden, attn_weights

# Testing the Models

encoder_test = EncoderRNN(10, 10, 2)
decoder_test = AttnDecoderRNN('general',10,10,2)
print(encoder_test)
print(decoder_test)

encoder_hidden = encoder_test.init_hidden()
word_input = Variable(torch.LongTensor([1,2,3]))
if USE_CUDA:
    encoder_test.cuda()
    word_input = word_input.cuda()
encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

word_inputs = Variable(torch.LongTensor([1,2,3]))
decoder_attns = torch.zeros(1,3,3)
decoder_hidden = encoder_hidden
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

if USE_CUDA:
    decoder_test.cuda()
    word_inputs = word_inputs.cuda()
    decoder_context = decoder_context.cuda()
for i in range(3):
    decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, encoder_hidden, encoder_outputs)
    print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
    decoder_attns[0,1] = decoder_attn.squeeze(0).cpu().data

# Training

teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, criterion, max_length=max_length):

    # zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_varible.size()[0]

    # run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[sos_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ration
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_contex, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_contex, decoder_hidden, encoder_outputs)

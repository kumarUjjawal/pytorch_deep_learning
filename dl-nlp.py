# Deep Learning for Natural Language Processing Using Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

# Introduction ot Pytorch's tensor library

# creating tensors

v_data = [1.,2.,3.]
v = torch.Tensor(v_data)

# create a matrix
m_data = [[1.,3.,2.],[4.,5.,6.]]
m = torch.Tensor(m_data)

# create a 3D tensor of size 2x2x2.
t_data = [[[1.,2.], [3.,4.]],
          [[5.,6.], [7.,8.]]]
t = torch.Tensor(t_data)

# tensor with random data and specified dimensionality

x = torch.randn((3,4,5))

# operation with tensors

x = torch.Tensor([1.,2.,3.])
y = torch.Tensor([4.,5.,6.])

z = x + y

# concatenation

z_1 = torch.cat([x,y])

# 2. Computation graphs and Automatic Differentiation

x = autograd.Variable(torch.Tensor([1.,2.,3.]), requires_grad=True)

y = autograd.Vaiable(torch.Tensor([4.,5.,6.]), requires_grad=True)

z = x + y

# sum up all the entries in z

s = z.sum()

s.backward() # calling .backard() on any variable will run backprop, starting from it.

# 3. Affine maps, Non-linearities and Objectives

# affine maps

lin = nn.Linear(4,2)
data = autograd.Variable(torch.randn(2,4))

# non-linearities

data = autograd.Variable(torch.randn(2,2))
print data
print F.relu(data)

# 4. Creating network components in Pytorch

# logistic regression bag-of-words classifier

data = [ ("me gusta comer en la cafeteria".split(), "SPANISH"),
                 ("Give it to me".split(), "ENGLISH"),
                          ("No creo que sea una buena idea".split(), "SPANISH"),
                                   ("No it is not a good idea to get lost at sea".split(), "ENGLISH") ]

test_data = [ ("Yo creo que si".split(), "SPANISH"),
                      ("it is lost on me".split(), "ENGLISH")]

# map each word in the vocab to a unique initeger,
# which will be its index into the bag of words vector

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

vocab_size = len(word_to_ix)
num_labels = 2

class BOWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BOWClassifier, self).__init__()

        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec))

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_toix[word]] += 1
    return vec.view(1,-1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

model = BOWClassifier(num_labels, vocab_size)

for param in model.parameters():
    print param

# pass in bow-vector wrapped in autograd.Variable

sample = data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

# run on test data before training
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)
print next(model.parameters())[:,word_to_ix["cero"]]

loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in xrange(100):
    for instance,label in data:
        model.zero_grad()
        
        bow_vec = autograd.Variable(make_bow_vector(label, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # forward pass
        log_probs = model(bow_vec)

        # compute loss
        loss = loss_fn(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)

    print log_probs
print next(model.parameters())[:,word_to_ix["cero"]]

# 5. Word Embeddings: Encoding Lexical Sementics

# example: n-gram language modeling

context_size = 2
embed_dim = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [ ([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in xrange(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, voacab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), embed_dim, context_size)

optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in xrange(10):
    total_loss = torch.Tensor([10])
    for context, target in trigrams:
        context_idxs = map(lambda w: word_to_ix[w], context)
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        model.zero_grad()

        log_probs = model(context_var)

        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

        loss.backward() # backward pass
        optimizer.step() # update the gradients
    losses.append(total_loss)
print losses

# computing word embeddings: continuous bag-of-words

context_size = 2
raw_text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()


word_to_ix = { word: i for i, word in enumerate(set(raw_text)) }
data = []
for i in xrange(2, len(raw_text) - 2):
    context = [ raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2] ]

    target = raw_text[i]
    data.append((context,target))

class CBOW(nn.Module):
    def __init__(self):
        pass
    def forward(self, inputs):
        pass

def make_context_vector(context, word_to_ix):
    idxs = map(lambda w: word_to_ix[w], context)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# 6. Sequence Models and Long-Short Term Memory Networks

lstm = nn.LSTM(3,3) # input_dim, output_dim
inputs = [ autograd.Variable(torch.randn((1,3))) for _ in xrange(5)]

hidden =  (autograd.Varible(torch.randn(1,1,3)), autograd.Variable(torch.randn((1,1,3))))

for i in inputs:
    out, hidden = lstm(i.view(1,1,-1),hidden)

inputs = torch.cat(inputs).view(len(inputs),1,-1)
hidden = (autograd.Variable(torch.randn(1,1,3)), autgrad.Variable(torch.randn((1,1,3))))

out,hidden = lstm(inputs, hidden)

# example: an lstm for part-of-speech tagging

def prepare_sequence(seq, to_ix):
    idxs = map(lambda w: to_ix[w], seq)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)



training_data = [
            ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
                ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

embed_dim = 6
hidden_dim = 6

class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim  = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1,1, self.hidden_dim)),autograd.Variable(torch.zeros(1,1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence),1,-1),self.hidden)

        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

model = LSTMTagger(embed_dim, hidden_dim, len(word_to_ix), len(tag_to_ix))

loss_function = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)

for epoch in xrange(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)

        loss = loss_functions(tag_scores, targets)
        loss.backward()
        optimizer.step()

# scores after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)

# 7. Dynamic toolkits, dynamic programming and the BiLSTM-CRF

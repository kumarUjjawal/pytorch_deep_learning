# word2vec implementation with skip-gram

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=True)

    for in in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]]) # target
        random_labels.append(skip_grams[i][1]) # context word

    return random_inputs, random_labels

# Model

class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()

        self.W = nn.Linear(voc_size, embedding_size, bias=False)
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        hidden_layer = self.W(X)
        output_layer = self.WT(hidden_layer)
        return output_layer

if __name__ == '__main__':
    batch_size = 2
    embedding_size = 2

    sentences = ['king queen state', 'troop army station', 'country state province']

    word_seq = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i,w in enumerate(word_list)}
    voc_size = len(word_list)

    # skip-gram model
    skip_grams = []
    for i in range(1, len(word_seq) - 1):
        target = word_dict([word_seq[i]])
        context = [word_dict[word_seq[i - 1]], word_dict[word_seq[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters, lr=0.001)

    # training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x,y)
        plt.annotate(label, xy=(x,y), xytext=(5,2), textcoord='offset points', ha='right', va='bottom')
    plt.show()

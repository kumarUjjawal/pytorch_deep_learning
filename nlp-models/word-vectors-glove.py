# Exploring word vectors with GloVe

import torch
import torchtext.vocab as vocab

# loading word vectors
glove = vocab.GloVe(name='6B', dim=100)

def get_word(word):
    return glove.vectors[glove.stoi[word]]

# finding closest vectors

def closest(vec, n=10):
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in glove.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

# function to print (word, distance) tuple pairs
def print_tuples(tuples):
    for tuple in tuples:
        print('(%.4f) %s' % (tuple[1], tuple[0]))


# word analogies with vector arithmetic
def analogy(w1,w2,w3, n=5, filter_given=True):
    closest_word = closest(get_word(w2) - get_word(w1) + get_word(w3))

    if filter_given:
        closeses_words = [t for t in colsest_words if t[0] not in [w1, w2, w3]]

    print_tuples(closest_words[:n])



"""
Pretraining word2vec using PyTorch and d2l library.

Source:
@book{zhang2020dive,
    title={Dive into Deep Learning},
        author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
            note={\url{https://d2l.ai}},
                year={2020}
                 }
"""

from d2l import torch as d2l
import torch
import os
import random
import random

# Reading and Preprocessing the Dataset

d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                                              '319d85e578af0cdc590547f26231e4e31cdf1e42')

def read_ptb():
    data_dir = d2l.download_extract('ptb')
    with open(os.path.join(data_dir, 'ptb.train.text')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()

# build vocab with words appeared not greater than 10 times
vocab = d2l.Vocab(sentences, min_freq=10)

# Subsampling

def subsampling(sentences, vocab):
    # map low frequency words inot <unk>
    sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line] for line in sentences]

    # count the frequency for each word
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # return true if to keep this token during subsampling
    def keep(token):
        return (random.uniform(0,1) < math.sqrt(1e-4 / counter[token] * num_tokens))

    # do the subsampling
    return [[tk for tk in line if keep(tk)] for line in sentences]

subsampled = subsampling(sentences, vocab)

# compare the sequence lengths before and after sampling

d2l.set_figure()
d2l.plt.hist([[len(line) for line in sentences],
              [len(line) for line in subsampled]])
d2l.plt.xlabel('# token per sentence')
d2l.plt.ylabel('count')
d2l.plt.legend(['origin','subsampled'])

# Loading the Dataset

# extract all the central target words and their context words

def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [],[]
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # exclude the central target word from the context words

            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

# Negative Sampling

class RandomGenerator:
    def __init__(self, sampling_weights):
        self.population = list(range(len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(self.population,
                                             self.sampling_weights,
                                             k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i-1]

generator = RandomGenerator([2,3,4])
[generator.draw() for _ in range(10)]

def get_negatives(all_contexts, corpus, k):
    counter = d2l.counter_corpus(corpus)
    sampling_weights = [counter[i] ** 0.75 for i in range(len(counter))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * k:
            neg = generator.draw()
            # noise words cannot be context words
            if neg not in contexts:
                negative.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, corpus, 5)

# Reading into Batches

def batchify(data):
    max_len = max(len(c) + len(n) for _,c,n in data)
    centers, contexts_negatives, masks, labels = [],[],[],[]
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_lne)]
        masks += [[1] * cur_len + [0] * (max_len - len(context))]
    return (d2l.reshape(torch.tensor(centers), (-1,1)), torch.tensor(contexts_negatives), torch.tensor(masks), torch.tensor(labels))

# Puttings All Things Together

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    num_workers = d2l.get_dataloader_worders()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)

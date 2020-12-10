# Bidirectional LSTM condidional random field for named-entity recognition

import torch
import torch.nn as nn
import torch.nn.fuctional as F
import torch.optim as optim
import torch.autograd as autograd

torch.malual_seed(1)

# helper function
def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    _,idx = torch.max(vec,1)
    return to_scalar(idx)

# compute log sum exp in a numerically stable way for the forward alogorithm

def log_sum_exp(vec):
    max_score = vec[0,argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim/2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transition.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim)), autograd.Variable(torch.randn(2,1, self.hidden_dim)))

    def _forward_alg(self, feats):
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        forward_var = autograd.Variable(init_alphas)
        
        for feat in feats:
            alphas_t = []
            for next_tag in xrange(self.tagset_size):
                emit_score = feat[next_tag].view(1,-1).expand(1,self.tagset_size)



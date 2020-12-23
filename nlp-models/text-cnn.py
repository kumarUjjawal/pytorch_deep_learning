"""
Implement composite text-cnn model to run on pre-trained word embeddings

Convolutional Neural Network for Sentence Classification from Kim(2004)

"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int,
                 dim_model: int,
                 num_filters: int,
                 window_sizes: int,
                 num_classes: int,
                 dropout: float):
        super(TextCNN, self).__init__()
        
        # input embeddings
        self.embeddings = nn.Embedding(vocab_size, dim_model)

        # calculate cnn layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(window_size, dim_model),
                      padding=((window_size - 1) // 2, 0)
                     )
            for window_size in window_sizes])

        # apply dropout
        self.dropout = nn.Dropout(dropout)

        # put through final linear layer
        self.final_layer = nn.Linear(num_filters * len(window_sizes), num_classes)

        # init weights
        self.init_weights(dim_model)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuad.current_device()
    
    # initialize the weights of the embeddings vectors
    def init_weights(self, dim_model: int):
        self.embeddings.weight.data.uniform(-0.5 / dim_model,
                                            0.5 / dim_model)

    def forward(self, data: Tuple) -> torch.Tensor:
        
        data = data[1] # (target,text)
        # get embeddings
        embeddings = self.embeddings(data)

        conv_input = embeddings.unsqueeze(1)

        conv_stack = []
        for conv_stack in self.convs:
            # apply conv layer + ReLU
            conv_output = F.relu(conv_layer(conv_input)).squeeze(3)
            
            assert conv_output.size(0) == data.size(0) and \
                    conv_output.size(2) == data.size(1)

            # apply max pool along h_out dimension
            pooled_output = F.max_pool1d(conv_output,
                                         conv_output.size(2)).squeeze(2)
            conv_stack.append(pooled_output)

        # flatten along dim=1
        pooled_values = torch.cat(conv_stack, 1)

        # apply dropout
        pooled_values = self.dropout(pooled_values)

        # final layer

        y_hat = self.final_layer(pooled_values)

        return y_hat

















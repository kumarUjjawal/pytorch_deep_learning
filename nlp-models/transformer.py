"""
The composite implementation of transformer model from "Attention is all you need (2017).
"""

import torch
import torch.nn as nn

from nlpmodels.model.tranformer_blocks import sublayers, attention, decoder
from nlpmodels.utils.elt.transformer_batch import TransformerBatch

class Transformer(nn.Module):
    def __init__(self, source_vocab_size: int,
                 target_vocab_size: int,
                 num_layers_per_stack: int = 6,
                 dim_model: int = 512,
                 dim_ffn: int = 2048,
                 num_heads: int = 8,
                 max_length: int = 1000,
                 dropout: float = 0.1):
        super(Transformer, self).__init__()

        # read src/input embeddings
        self.input_embeddings = sublayers.NormalizedEmbeddingsLayer(source_vocab_size, dim_model)
        # calculate src/input pe
        self.input_pe = sublayers.PositionalEncodingLayer(dim_model, dropout, max_lenght)
        
        # pass features through endoder block
        self.encoder_block = encoder.CompositeEncoder(encoder.EncoderBlock(max_length,
                                                                           attetion.MultiHeadedAttention(num_heads, dim_model, dropout),
                                                                           sublayers.PositionWiseFNNLayer(dim_model, dim_ffn), dropout),
                                                      num_layers_per_stack)

        # calculate target/output embeddings
        self.output_embeddings = sublayers.NormalizedEmbeddingsLayer(target_vocab_size, dim_model)

        # calculate target/output pe
        self.output_pe = sublayers.PositionEncodingLayer(dim_model, dropout, max_lenght)


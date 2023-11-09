import torch.nn as nn

from .conv_layers import ConvNormAct, FeedForwardNetwork
from .rnn_layers import RNNProjection, GlobalAttentionRNN
from .fusion import InjectionMultiSum
from .attention import GlobalAttention, MultiHeadSelfAttention


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)

        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))

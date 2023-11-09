import torch
import torch.nn as nn


class RNNProjection(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "LSTM",
        dropout: float = 0,
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(RNNProjection, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.num_direction = int(bidirectional) + 1

        self.norm1 = nn.LayerNorm(self.input_size)
        self.rnn = getattr(nn, rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.proj = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * self.num_direction, self.input_size),
            nn.Dropout(self.dropout),
        )
        self.norm2 = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor):
        res = x
        x = x.transpose(1, 2).contiguous()

        x = self.norm1(x)
        residual = x
        self.rnn.flatten_parameters()
        x = self.rnn(x)[0]  # B, L, num_direction * H
        x = self.proj(x)
        x = self.norm2(x + residual)  # B, L, N

        x = x.transpose(1, 2).contiguous()
        x = x + res
        return x


class GlobalAttentionRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        dropout: float = 0.1,
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalAttentionRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else self.in_chan
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.RNN = RNNProjection(self.in_chan, self.hid_chan, self.rnn_type, self.dropout, self.bidirectional)

    def forward(self, x: torch.Tensor):
        x = self.RNN(x)
        return x

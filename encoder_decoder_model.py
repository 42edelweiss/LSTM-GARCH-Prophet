import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # You don't need view() if x already has the right shape,
        # but keeping it is OK if you're sure x is contiguous.
        x = x.view(x.shape[0], x.shape[1], self.input_size)

        out, (h_n, c_n) = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        # h_n, c_n: (num_layers, batch, hidden_size)

        return out, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        """
        x: (batch, input_size)  OR (batch, 1, input_size)
        h: (h_n, c_n) each (num_layers, batch, hidden_size)
        """
        if x.dim() == 2:
            # (batch, input_size) -> (batch, 1, input_size)
            x = x.unsqueeze(1)

        out, h = self.lstm(x, h)
        # out: (batch, 1, hidden_size)

        y = self.linear(out[:, 0, :])   # (batch, output_size)
        return y, h

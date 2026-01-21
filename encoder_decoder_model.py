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
        out, (h_n, c_n) = self.lstm(x)
        return out, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        """
        x: (batch, input_size) or (batch, 1, input_size)
        h: (h_n, c_n) each (num_layers, batch, hidden_size)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> (batch, 1, input_size)

        out, h = self.lstm(x, h)          # out: (batch, 1, hidden)
        y = self.linear(out[:, 0, :])     # (batch, output_size)
        return y, h


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

    def train_model(
        self,
        train,              # (batch, input_len, input_size)
        target,             # (batch, target_len, output_size)
        epochs,
        method="recursive", # "recursive", "teacher_forcing", "mixed_teacher_forcing"
        tfr=0.5,
        lr=0.01,
        dynamic_tf=False,
        print_every=10
    ):
        device = next(self.parameters()).device
        train = train.to(device)
        target = target.to(device)

        batch_size = train.shape[0]
        target_len = target.shape[1]
        output_size = target.shape[2]

        losses = np.full(epochs, np.nan, dtype=np.float32)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for e in range(epochs):
            self.train()
            optimizer.zero_grad()

            # Encode
            _, enc_h = self.encoder(train)

            # First decoder input: last element of encoder input sequence
            dec_in = train[:, -1, :]     # (batch, input_size)
            dec_h = enc_h

            predicted = torch.zeros(batch_size, target_len, output_size, device=device)

            if method == "recursive":
                for t in range(target_len):
                    dec_out, dec_h = self.decoder(dec_in, dec_h)  # (batch, output_size)
                    predicted[:, t, :] = dec_out
                    # feed back prediction
                    dec_in = dec_out

            elif method == "teacher_forcing":
                use_tf = (random.random() < tfr)
                for t in range(target_len):
                    dec_out, dec_h = self.decoder(dec_in, dec_h)
                    predicted[:, t, :] = dec_out
                    dec_in = target[:, t, :] if use_tf else dec_out

            elif method == "mixed_teacher_forcing":
                for t in range(target_len):
                    dec_out, dec_h = self.decoder(dec_in, dec_h)
                    predicted[:, t, :] = dec_out
                    dec_in = target[:, t, :] if (random.random() < tfr) else dec_out
            else:
                raise ValueError(f"Unknown method: {method}")

            loss = criterion(predicted, target)
            loss.backward()
            optimizer.step()

            losses[e] = loss.item()

            if (e % print_every) == 0:
                print(f"Epoch {e}/{epochs}: {loss.item():.6f}")

            if dynamic_tf and tfr > 0:
                tfr = max(0.0, tfr - 0.02)

        return losses

    @torch.no_grad()
    def predict(self, x, target_len):
        """
        x: (batch, input_len, input_size)
        returns y: (batch, target_len, output_size)
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)

        batch_size = x.shape[0]
        output_size = self.decoder.linear.out_features

        _, enc_h = self.encoder(x)
        dec_in = x[:, -1, :]   # (batch, input_size)
        dec_h = enc_h

        y = torch.zeros(batch_size, target_len, output_size, device=device)

        for t in range(target_len):
            dec_out, dec_h = self.decoder(dec_in, dec_h)
            y[:, t, :] = dec_out
            dec_in = dec_out

        return y

import torch
from torch import nn


class LSTM_Generator(nn.Module):
    def __init__(self, z_dim, polymer_dim):
        super(LSTM_Generator, self).__init__()

        self.rnn = nn.LSTM(
            input_size=z_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, polymer_dim)

    def forward(self, z):
        r_out, (h_n, h_c) = self.rnn(z, None)
        out = self.out(r_out[:, -1, :])
        return out


class LSTM_Discriminator(nn.Module):
    def __init__(self, polymer_dim):
        super(LSTM_Discriminator, self).__init__()

        self.rnn = nn.LSTM(
            input_size=polymer_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 1)

    def forward(self, z):
        r_out, (h_n, h_c) = self.rnn(z, None)
        out = self.out(r_out[:, -1, :])
        return out


def get_noise_discrete(n_sample, noise_dim, device='cpu'):
    return torch.randint(0, 4, size=(n_sample, noise_dim, 1), device=device) / 4


rnn_gen = LSTM_Generator(8, 14)
noise = get_noise_discrete(1, 8)
print(noise)
print(noise.size())
output = rnn_gen(noise)

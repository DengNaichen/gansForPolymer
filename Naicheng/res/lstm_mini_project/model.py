import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    """
    A two hidden-layer discriminative neural network
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(14, 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(16, 32),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(32, 64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        # self.hidden3 = nn.Sequential(
        #     nn.Linear(64, 16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Dropout(0.3)
        # )
        self.out = nn.Sequential(
            torch.nn.Linear(64, 2),
            # torch.nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        # z = self.hidden3(z)
        z = self.out(z)
        return z


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 2)

    def forward(self, z):
        r_out, (h_n, h_c) = self.rnn(z, None)
        out = self.out(r_out[:, -1, :])
        return out

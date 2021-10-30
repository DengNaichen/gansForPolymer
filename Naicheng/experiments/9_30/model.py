import torch.nn as nn
import torch


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 8
        n_out = 31

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(inplace=True),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(64, n_out),  # double
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.hidden3(z)
        z = self.out(z)
        return z


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 31
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64,128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(64, n_out)
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.hidden3(z)
        z = self.out(z)
        return z

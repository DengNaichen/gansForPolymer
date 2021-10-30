import torch.nn as nn
import torch
import numpy as np


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 45
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(32, n_out)
            # torch.nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.out(z)
        return z


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 8
        n_out = 45

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(inplace=True),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(64, n_out),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.out(z)
        return z


# Noise
def noise(size):
    n = torch.tensor(np.random.normal(0, 1, (size, 8)), dtype=torch.float32)
    return n
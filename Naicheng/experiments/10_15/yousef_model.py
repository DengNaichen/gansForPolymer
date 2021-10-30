import torch.nn as nn
import torch
import numpy as np


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 14
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 128), # yousef, best result
            # nn.Linear(n_features, 64), # half
            # nn.Linear(n_features, 256),  # double
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 64), # yousef
            # nn.Linear(64, 32), # half
            # nn.Linear(256, 128), # double
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32), # yousef
            # nn.Linear(32, 16), # half
            # nn.Linear(128, 64),  # double
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(32, n_out) # yousef
            # nn.Linear(16, n_out) # half
            # torch.nn.Linear(64, n_out) # double
            # torch.nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.out(z)
        return z


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 8
        n_out = 14

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 16), # yousef
            # nn.Linear(n_features, 8), # half
            # nn.Linear(n_features, 32),  # double
            nn.ReLU(inplace=True),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(16, 64), # yousef
            # nn.Linear(8, 32),
            # nn.Linear(32, 128), #double
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32), # yousef
            # nn.Linear(32, 16), # half
            # nn.Linear(128, 64), #double
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(32, n_out), # yousef
            # nn.Linear(16, n_out), # half
            # nn.Linear(64, n_out),  # double
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.out(z)
        return z


# Noise
def noise(size):

    n = torch.tensor(np.random.normal(0, 1, (size, 8)), dtype=torch.float32)

    return n
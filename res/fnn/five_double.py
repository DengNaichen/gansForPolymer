import torch.nn as nn
import torch


class GeneratorNet(torch.nn.Module):
    """
    A two hidden-layer generative neural network
    """
    def __init__(self, z_dim, polymer_dim):
        super(GeneratorNet, self).__init__()

        self.hidden0 = nn.Sequential(
            # nn.Linear(n_features, 128),
            nn.Linear(z_dim, 256),
            nn.ReLU(inplace=True),
            # nn.SiLU(inplace=True),
            # nn.Dropout(0.1)
        )
        self.hidden1 = nn.Sequential(
            # nn.Linear(128, 64),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.SiLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            # nn.Linear(64, 32),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.SiLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            # nn.Linear(32, n_out),
            nn.Linear(64, polymer_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.out(z)
        return z


class DiscriminatorNet(torch.nn.Module):
    """
    A two hidden-layer discriminative neural network
    """

    def __init__(self, polymer_dim):
        super(DiscriminatorNet, self).__init__()

        self.hidden0 = nn.Sequential(
            # nn.Linear(n_features, 128),
            nn.Linear(polymer_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            # nn.Linear(128, 64), # original
            nn.Linear(256, 128),  # double
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            # nn.Linear(64, 32),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            # torch.nn.Linear(32, n_out)
            torch.nn.Linear(64, 1)
            # torch.nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.out(z)
        return z

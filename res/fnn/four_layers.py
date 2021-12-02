import torch.nn as nn
import torch


class GeneratorNet(torch.nn.Module):
    """
    A two hidden-layer generative neural network
    """
    def __init__(self, z_dim, polymer_dim):
        super(GeneratorNet, self).__init__()

        self.hidden0 = nn.Sequential(
            # nn.Linear(z_dim, 128),
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid(),
        )
        self.hidden1 = nn.Sequential(
            # nn.Linear(128, 64),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid(),
            # nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            # nn.Linear(64, polymer_dim),
            nn.Linear(128, polymer_dim),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        # print(z)
        z = self.hidden1(z)
        # print(z)
        z = self.out(z)
        # print(z)
        return z


class DiscriminatorNet(torch.nn.Module):
    """
    A two hidden-layer discriminative neural network
    """
    def __init__(self, polymer_dim):
        super(DiscriminatorNet, self).__init__()

        self.hidden0 = nn.Sequential(
            # nn.Linear(polymer_dim, 128),
            nn.Linear(polymer_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden1 = nn.Sequential(
            # nn.Linear(128, 64),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            # torch.nn.Linear(64, 1),
            torch.nn.Linear(128, 1),
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.out(z)
        return z

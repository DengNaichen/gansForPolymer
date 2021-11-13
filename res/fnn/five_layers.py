import torch.nn as nn
import torch


class GeneratorNet(torch.nn.Module):
    """
    A two hidden-layer generative neural network
    """

    def __init__(self, z_dim, polymer_dim):
        super(GeneratorNet, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(32, polymer_dim),
            nn.BatchNorm1d(polymer_dim),
            # nn.Sigmoid()
            nn.Tanh()
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
            nn.Linear(polymer_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(32, 1)
            # don't need sigmoid here since sigmoid is a build-in function for
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.out(z)
        return z

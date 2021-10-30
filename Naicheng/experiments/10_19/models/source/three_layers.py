import torch.nn as nn
import torch


class DiscriminatorNet(torch.nn.Module):
    """
    A two hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 14
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        # self.hidden1 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.2)
        # )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.2)
        # )
        self.out = nn.Sequential(
            torch.nn.Linear(128, n_out)
            # torch.nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        # z = self.hidden1(z)
        # z = self.hidden2(z)
        z = self.out(z)
        return z


class GeneratorNet(torch.nn.Module):
    """
    A two hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 8
        n_out = 14

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(inplace=True),
            # nn.SiLU(inplace=True),
            # nn.Dropout(0.1)
        )
        # self.hidden1 = nn.Sequential(
        #     nn.Linear(16, 64),
        #     nn.ReLU(inplace=True),
        #     # nn.SiLU(inplace=True),
        #     nn.Dropout(0.3)
        # )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU(inplace=True),
        #     # nn.SiLU(inplace=True),
        #     nn.Dropout(0.2)
        # )
        self.out = nn.Sequential(
            nn.Linear(128, n_out),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        # z = self.hidden1(z)
        # z = self.hidden2(z)
        z = self.out(z)
        return z

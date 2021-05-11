from torch import nn


class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: dimension of input noise
        im_dim: dimension of generator output
        hidden_dim: the hidden dimension, a scalar
    """

    def __init__(self, z_dim=4, im_dim=15, hidden_dim=4):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input layer
            self.get_generator_block(z_dim, hidden_dim),
            # hidden layers
            self.get_generator_block(hidden_dim, hidden_dim * 2),
            self.get_generator_block(hidden_dim * 2, hidden_dim * 4),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 8),
            # output layer
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def get_generator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        """
        Returns:
            the sequential model
        """
        return self.gen

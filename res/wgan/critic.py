from torch import nn


class Critic(nn.Module):

    def __init__(self, im_dim=15, hidden_dim=4):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            self.get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def get_discriminator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc

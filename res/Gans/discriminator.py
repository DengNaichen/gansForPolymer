from torch import nn


class Discriminator(nn.Module):

    def __init__(self, im_dim=15, hidden_dim=4):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input layer, input is the image tensor, output is the input of the next hidden layer
            # here the input dimension is 784 * 512, where 784 is the 28 * 28 which is the size of image，
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            # hidden layer, activation function for each layer is leak relu.
            self.get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.get_discriminator_block(hidden_dim * 2, hidden_dim),
            # output layer, output is 1, which is a probability of real or fake.
            nn.Linear(hidden_dim, 1),
            # todo: why here is no activation function? I think a sigmoid function should be here?
            # result, the sigmoid function is a build function for nn.BCEWithLogitsLoss(), so we don't need it here,
            # But theoretically, a activation function(sigmoid) are required.
        )

    def get_discriminator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # see detail at:
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
            # for a single layer without activation function.
            nn.LeakyReLU(0.2, inplace=True)
            # nn.Tanh()
            # slope for x < 0 is 0.2
            # inplace = True: inplace=True means that it will modify the input directly, without allocating any
            # additional output. It can sometimes slightly decrease the memory usage.
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc

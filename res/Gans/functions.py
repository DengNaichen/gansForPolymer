import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 60, 1)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, noise_dim, device='cpu'):
    """
    Returns a tensor filled with random numbers from a normal distribution
    with mean 0 and variance 1 (also called the standard normal distribution).
    n_sample * noise_dim
    """
    
    return torch.randn(n_samples, noise_dim, device=device)


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    """
    :param gen:
    :param disc:
    :param criterion:
    :param num_images:
    :param z_dim:
    :param device:
    :return:
    todo：why we need get generator loss? or what generator loss mean?
    """
    # num_image * z_dim matrix
    fake_noise = get_noise(num_images, z_dim, device=device)
    # fake is the output image
    fake = gen(fake_noise)
    # disc_fake_pred is the output of discriminator, or the prediction of discriminator, should be scalar? I think so
    disc_fake_pred = disc(fake)
    # todo: figure out the dimension
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    return gen_loss


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    :param gen: the generator nn
    :param disc: the discriminator nn
    :param criterion: loss function, here I use BCE loss function, see detail at:

    :param real: todo
    :param num_images: number of sample
    :param z_dim: noise
    :param device: cpu for my laptop
    :return:
    """
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise)
    # todo: why need detach() here?
    # since the generator is needed when calculating the discriminator's loss,
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    # You need a 'ground truth' tensor in order to calculate the loss.
    # the ground truth of fake is zero
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    # the ground truth of real is one
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    # the loss is average loss of fake and real

    return disc_loss





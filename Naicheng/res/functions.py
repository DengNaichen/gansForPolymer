import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn as nn


# get noise
def get_noise(n_samples, noise_dim, device='cpu'):
    return torch.randn(n_samples, noise_dim, device=device)


def get_noise_uniform(n_samples, noise_dim, device='cpu'):
    return torch.rand(n_samples, noise_dim, device=device)


def get_noise_discrete(n_sample, noise_dim, device='cpu'):
    return torch.randint(0, 4, size=(n_sample, noise_dim), device=device) / 4


# get discriminator loss
def get_disc_loss(gen, disc, loss_func, real, num_images, z_dim, noise_type, device, c_lambda=None):

    global noise
    assert noise_type == 'normal' or 'discrete', 'noise type should be normal or discrete'

    if noise_type == 'normal':
        noise = get_noise(num_images, z_dim, device=device)
    elif noise_type == 'discrete':
        noise = get_noise_discrete(num_images, z_dim, device)

    fake = gen(noise)
    disc_fake_pred = disc(fake.detach())
    disc_real_pred = disc(real)

    if loss_func == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        disc_loss = __disc_bce_loss(disc_fake_pred, disc_real_pred, criterion)

    # elif loss_func == 'wloss':
    #     disc_loss = __disc_get_wloss(real, device, disc, fake, disc_fake_pred, disc_real_pred, c_lambda)

    return disc_loss


# get BCE loss for discriminator
def __disc_bce_loss(disc_fake_pred, disc_real_pred, criterion):

    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


# # get w loss for discriminator
# def __disc_get_wloss(real, device, disc, fake, disc_fake_pred, disc_real_pred, c_lambda):
#     epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
#     gradient = __get_gradient(disc, real, fake.detach(), epsilon)
#     gp = __gradient_penalty(gradient)
#     disc_loss = torch.mean(disc_fake_pred) - torch.mean(disc_real_pred) + c_lambda * gp
#     return disc_loss


# for wgan, get the gradient penalty
# get gradient
def __get_gradient(crit, real, fake, epsilon):
    # note: the epsilon is an tensor contains random numbers
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


# get penalty for given gradient
def __gradient_penalty(gradient):
    # flatten the image, but my input is already flatten, so I this step
    # gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


# get generator loss
def get_gen_loss(gen, disc, loss_func, num_images, z_dim, noise_type, device):

    global noise
    assert noise_type == 'normal' or 'discrete', 'noise type should be normal or discrete'
    if noise_type == 'normal':
        noise = get_noise(num_images, z_dim, device=device)
    elif noise_type == 'discrete':
        noise = get_noise_discrete(num_images, z_dim, device)

    fake = gen(noise)
    disc_fake_pred = disc(fake)

    if loss_func == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    elif loss_func == 'wloss':
        gen_loss = -1. * torch.mean(disc_fake_pred)

    return gen_loss

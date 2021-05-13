from tqdm.auto import tqdm
import torch
import res.fnn.functions as func
import matplotlib.pyplot as plt

from res.fnn.generator import Generator
from res.fnn.discriminator import Discriminator


def training_bce(gen, disc, z_dim, n_epochs, dataloader, device, disc_opt, gen_opt,
                 display_step):

    mean_discriminator_loss = 0
    mean_generator_loss = 0
    cur_step = 0

    for epoch in range(n_epochs):

        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            # update discriminator
            disc_opt.zero_grad()
            disc_loss = func.get_disc_loss(gen, disc, "bce", real, cur_batch_size, z_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # update generator
            gen_opt.zero_grad()
            gen_loss = func.get_gen_loss(gen, disc, 'bce', cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            # todo
            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            # visualization data
            # todo
            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, "
                    f"discriminator loss: {mean_discriminator_loss}")
                fake_noise = func.get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1


def training_wloss(n_epochs, dataloader, device, disc_repeats, gen, gen_opt,
                   disc, disc_opt, z_dim, c_lambda, display_step):
    cur_step = 0
    generator_losses = []
    disc_losses = []

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            # real_image = real_image.to(device)
            real = real.view(cur_batch_size, -1).to(device)

            mean_iteration_disc_loss = 0
            # fix generator, update critic five times
            for _ in range(disc_repeats):
                # Update critic #
                disc_opt.zero_grad()
                disc_loss = func.get_disc_loss(gen, disc, "wloss", real, cur_batch_size, z_dim, device, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_disc_loss += disc_loss.item() / disc_repeats
                # Update gradients
                disc_loss.backward(retain_graph=True)
                # Update optimizer
                disc_opt.step()
            disc_losses += [mean_iteration_disc_loss]

            # Update generator #
            gen_opt.zero_grad()
            gen_loss = func.get_gen_loss(gen, disc, 'wloss', cur_batch_size, z_dim, device)
            gen_loss.backward()

            # Update the weights
            gen_opt.step()

            # keep track of the average generator loss
            generator_losses += [gen_loss.item()]

            # Visualization code #
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                disc_mean = sum(disc_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, Discriminator loss: {disc_mean}")
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(disc_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Discriminator Loss"
                )
                plt.legend()
                plt.show()

            cur_step += 1


def initialize_model(z_dim, im_dim, hidden_dim, device, lr, beta_1, beta_2):
    gen = Generator(z_dim, im_dim=im_dim, hidden_dim=hidden_dim).to(device)
    disc = Discriminator(im_dim=im_dim, hidden_dim=hidden_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    return gen, disc, gen_opt, disc_opt

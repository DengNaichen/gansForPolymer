from tqdm.auto import tqdm
import torch
import functions as func
import matplotlib.pyplot as plt


def training_bce(gen, disc, z_dim, n_epochs, dataloader, device, disc_opt, gen_opt,
                 display_step, noise_type, display=True):
    """

    """
    assert noise_type == 'normal' or 'discrete'

    gen_loss_list = []
    disc_loss_list = []

    mean_discriminator_loss = 0
    mean_generator_loss = 0
    cur_step = 0

    for epoch in range(n_epochs):

        for real, _ in dataloader:
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            # update discriminator
            disc_opt.zero_grad()
            disc_loss = func.get_disc_loss(gen, disc, "bce", real, cur_batch_size, z_dim, noise_type, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # update generator
            gen_opt.zero_grad()
            gen_loss = func.get_gen_loss(gen, disc, 'bce', cur_batch_size, z_dim, noise_type, device)
            gen_loss.backward()
            gen_opt.step()

            # todo
            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            # visualization data
            # todo
            if display is True:
                if cur_step % display_step == 0 and cur_step > 0:
                    # print(
                    #     f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, "
                    #     f"discriminator loss: {mean_discriminator_loss}")
                    # if noise_type == 'normal':
                    #     fake_noise = func.get_noise(cur_batch_size, z_dim, device=device)
                    # elif noise_type == 'discrete':
                    #     fake_noise = func.get_noise_discrete(cur_batch_size, z_dim, device=device)
                    #
                    # fake = gen(fake_noise)

                    disc_loss_list.append(mean_discriminator_loss)
                    gen_loss_list.append(mean_generator_loss)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
            cur_step += 1
    return disc_loss_list, gen_loss_list

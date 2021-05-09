from tqdm.auto import tqdm
import torch
import res.Gans.functions as func


def Training(gen, disc, criterion, z_dim, n_epochs, dataloader, device, disc_opt, gen_opt,
             cur_step, display_step, mean_discriminator_loss, mean_generator_loss):
    disc_loss_list = []
    gen_loss_list = []
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)
            # real = torch.stack(real).view(cur_batch_size, -1)
            # update discriminator
            disc_opt.zero_grad()
            disc_loss = func.get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # update generator
            gen_opt.zero_grad()
            gen_loss = func.get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step
            disc_loss_list.append(mean_discriminator_loss)
            gen_loss_list.append(mean_generator_loss)

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, "
                    f"discriminator loss: {mean_discriminator_loss}")
                fake_noise = func.get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                #                 func.show_tensor_images(fake)
                #                 func.show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1

    return disc_loss_list, gen_loss_list

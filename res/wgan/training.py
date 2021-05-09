from tqdm.auto import tqdm
from res.wgan.functions import get_noise, get_gradient, get_crit_loss, gradient_penalty, get_gen_loss
import torch


def Training(n_epochs, dataloader, device, crit_repeats, gen, gen_opt,
             crit, crit_opt, z_dim, c_lambda, display_step):
    cur_step = 0
    generator_losses = []
    critic_losses = []
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            # real_image = real_image.to(device)
            real = real.view(cur_batch_size, -1).to(device)

            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                # Update critic #
                crit_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]
            # Update generator #
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()

            # Update the weights
            gen_opt.step()

            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]

            # Visualization code #
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins

            cur_step += 1

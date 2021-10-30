import numpy as np
import torch
import json
from torch.utils.data import DataLoader
import model as model

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)

from res.process_data.dataset import tensor_dataset
from res.main import training_bce
import res.process_data.process_output as pro_out

directions = np.load('../../data/random/32monos/four_directions.npy')


print(directions[0])


shuffle = True
batch_size = 125
num_worker = 0
pin_memory = True
input_tensor = torch.Tensor(directions)
my_dataset = tensor_dataset(input_tensor, 31, 1)
my_dataloader = DataLoader(dataset= my_dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_worker,
                            pin_memory=pin_memory)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dim = 8
lr = 0.001
gen = model.GeneratorNet()
gen_opt = torch.optim.Adam(gen.parameters(), lr = lr)
disc = model.DiscriminatorNet()
disc_opt = torch.optim.Adam(disc.parameters(), lr = lr)

epoch_step = 100
display_step = 782
total_epoch = 0
loss_value_disc = {}
loss_value_gen = {}

for i in range(30):
    disc_loss, gen_loss = training_bce(gen, disc, z_dim, epoch_step, my_dataloader,
                                       device, disc_opt, gen_opt, display_step)
    total_epoch += epoch_step
    pro_out.save_model(gen, disc, 'models/model', total_epoch)
    loss_value_disc[f'epoch{total_epoch}'] = disc_loss
    loss_value_gen[f'epoch{total_epoch}'] = gen_loss

with open('gen_loss.json', 'w') as fp:
    json.dump(loss_value_gen, fp)
with open('disc_loss.json', 'w') as fp:
    json.dump(loss_value_disc, fp)
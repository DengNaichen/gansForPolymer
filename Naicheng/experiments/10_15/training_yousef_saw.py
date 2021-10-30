import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import yousef_model as model

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)

from res.process_data.dataset import tensor_dataset
from res.main import training_bce
import res.process_data.process_output as pro_out

directions = np.load('../../data/self_avoid/0_5_1/directions_14.npy')



shuffle = True
batch_size = 128
num_worker = 0
pin_memory = True
input_tensor = torch.Tensor(directions)
my_dataset = tensor_dataset(input_tensor, 14, 1)
my_dataloader = DataLoader(dataset= my_dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_worker,
                            pin_memory=pin_memory)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dim = 8
lr = 0.0001
gen = model.GeneratorNet()
gen_opt = torch.optim.Adam(gen.parameters(), lr = lr)
disc = model.DiscriminatorNet()
disc_opt = torch.optim.Adam(disc.parameters(), lr = lr)


epoch_step = 100
display_step = 782
total_epoch = 0
loss_value_disc = {}
loss_value_gen = {}

for i in range(20):
    disc_loss, gen_loss = training_bce(gen, disc, z_dim, epoch_step, my_dataloader,
                                       device, disc_opt, gen_opt, display_step)
    total_epoch += epoch_step
    pro_out.save_model(gen, disc, 'model/self_avoid/0_5_1/yousef_model', total_epoch)
    loss_value_disc[f'epoch{total_epoch}'] = disc_loss
    loss_value_gen[f'epoch{total_epoch}'] = gen_loss

with open('model/self_avoid/0_5_1/yousef_gen_loss_three_layers.json', 'w') as fp:
    json.dump(loss_value_gen, fp)
with open('model/self_avoid/0_5_1/yousef_disc_loss_three_layers.json', 'w') as fp:
    json.dump(loss_value_disc, fp)
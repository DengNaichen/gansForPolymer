import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import single_node as model

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)

from res.process_data.dataset import tensor_dataset
from res.main import training_bce
import res.process_data.process_output as pro_out

directions = np.load('../../data/random/16monos/four_directions.npy')

shuffle = True
batch_size = 125
num_worker = 0
pin_memory = True
input_tensor = torch.Tensor(directions)
my_dataset = tensor_dataset(input_tensor, 15, 1)
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

# load trained model
# gen_checkpoint = torch.load('models/model_gan_1000.pth.tar')
# disc_checkpoint = torch.load('models/model_disc_1000.pth.tar')
# gen.state_dict(gen_checkpoint['gen_state_dict'])
# gen.state_dict(disc_checkpoint['disc_state_dict'])

epoch_step = 100
display_step = 782
total_epoch = 0
loss_value_disc = {}
loss_value_gen = {}

for i in range(20):
    disc_loss, gen_loss = training_bce(gen, disc, z_dim, epoch_step, my_dataloader,
                                       device, disc_opt, gen_opt, display_step)
    total_epoch += epoch_step
    pro_out.save_model(gen, disc, 'random_models/single_node/model', total_epoch)
    loss_value_disc[f'epoch{total_epoch}'] = disc_loss
    loss_value_gen[f'epoch{total_epoch}'] = gen_loss

with open('random_gen_loss_single_node.json', 'w') as fp:
    json.dump(loss_value_gen, fp)
with open('random_disc_loss.json_single_node', 'w') as fp:
    json.dump(loss_value_disc, fp)
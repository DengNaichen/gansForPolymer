# import third-part modules
import argparse
import numpy as np
import json
import torch
from torch.utils.data import DataLoader

# import my models
import fnn.single_node as single_node
import fnn.three_layers as three_layers
import fnn.four_layers as four_layers
import fnn.five_layers as five_layers
import fnn.six_layers as six_layers

# import my modules
from process_data.dataset import tensor_dataset
from training import training_bce
import process_data.save_data as save_data

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training，the total'
                                                             'epoch is n_epoch * saving step')
parser.add_argument('--z_dim', type=int, default=8, help='dimension of input random noise')
parser.add_argument('--polymer_dim', type=int, default=14, help='dimension of fake polymers')
parser.add_argument('--saving_step', type=int, default=200, help='saving model per how many epoch')
parser.add_argument('--device', type=str, default='cpu', help='device, cpu or cuda')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

# load data and saving model
parser.add_argument('--input_data_path', type=str, help='path of data source: '
                                                        '../data/random or self_avoid')

parser.add_argument('--output_models_path', type=str, help='path of output models: '
                                                           'experiment/data/models '
                                                           'note: the last one is the model name')

parser.add_argument('--model', type=str, help='the input for the argument should be: '
                                              'single node, three layers, four layers, '
                                              'five layers or six layers')
parser.add_argument('--noise_type', type=str, help='normal or discrete')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    z_dim = opt.z_dim
    polymer_dim = opt.polymer_dim
    device = opt.device
    total_epoch = 0
    display_step = 782
    saving_step = opt.saving_step
    output_path = opt.output_models_path
    n_epochs = opt.n_epochs
    model = opt.model
    noise_type = opt.noise_type

    # load models
    if model == 'single node':
        gen = single_node.GeneratorNet(z_dim, polymer_dim)
        disc = single_node.DiscriminatorNet(polymer_dim)
    elif model == 'three layers':
        gen = three_layers.GeneratorNet(z_dim, polymer_dim)
        disc = three_layers.DiscriminatorNet(polymer_dim)
    elif model == 'four layers':
        gen = four_layers.GeneratorNet(z_dim, polymer_dim)
        disc = four_layers.DiscriminatorNet(polymer_dim)
    elif model == 'five layers':
        gen = five_layers.GeneratorNet(z_dim, polymer_dim)
        disc = five_layers.DiscriminatorNet(polymer_dim)
    elif model == 'six layers':
        gen = six_layers.GeneratorNet(z_dim, polymer_dim)
        disc = six_layers.DiscriminatorNet(polymer_dim)

    # load optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr=opt.lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=opt.lr)

    # load input data
    directions = np.load(opt.input_data_path)
    lengthOfPolymer = np.shape(directions)[1]

    shuffle = True
    batch_size = 125
    num_worker = 0
    pin_memory = True
    input_tensor = torch.Tensor(directions)
    my_dataset = tensor_dataset(input_tensor, lengthOfPolymer, 1)
    my_dataloader = DataLoader(dataset=my_dataset,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               num_workers=num_worker,
                               pin_memory=pin_memory)

    loss_value_disc = {}
    loss_value_gen = {}
    for i in range(n_epochs):
        disc_loss, gen_loss = training_bce(gen, disc, z_dim, saving_step, my_dataloader,
                                           device, disc_opt, gen_opt, noise_type, display_step)
        total_epoch += saving_step
        save_data.save_model(gen, disc, output_path, total_epoch)
        loss_value_disc[f'epoch{total_epoch}'] = disc_loss
        loss_value_gen[f'epoch{total_epoch}'] = gen_loss

    save_data.save_loss(loss_value_gen, loss_value_disc, output_path)

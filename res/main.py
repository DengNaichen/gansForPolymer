# import third-part modules
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
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
from training import training_bce, training_w_loss_gp
import process_data.save_data as save_data
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training，the total'
                                                             'epoch is n_epoch * saving step')
parser.add_argument('--z_dim', type=int, default=8, help='dimension of input random noise')
parser.add_argument('--polymer_dim', type=int, default=16, help='dimension of fake polymers, 16, 32, or 64')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--saving_step', type=int, default=200, help='saving model per how many epoch')
parser.add_argument('--load', type=bool, default=False, help='if load a trained model')
parser.add_argument('--load_path_gen', type=str, default='', help='if load a trained model, the generator path of load')
parser.add_argument('--load_path_disc', type=str, default='', help='if load a trained model, the disc path of load')
parser.add_argument('--device', type=str, default='cpu', help='device, cpu or cuda')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

# load data and saving model

parser.add_argument('--output_models_path', type=str, help='path of output models: '
                                                           'experiment/data/models '
                                                           'note: the last one is the model name')

parser.add_argument('--model', type=str, help='the input for the argument should be: '
                                              'single node, three layers, four layers, '
                                              'five layers or six layers')
parser.add_argument('--noise_type', type=str, help='normal or discrete')
parser.add_argument('--total_epoch', type=int, default=0, help='total_epoch')
# parser.add_argument('--encoding_way', type=str, default='encoding method',
#                     help='the encoding method, can be directions or sin_cos')
parser.add_argument('--data_type', type=str, help='on_lattices or off_lattices')
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    # load numpy array
    data_type = opt.data_type
    polymer_length = opt.polymer_dim

    directions = np.load('../data/random/' + data_type + f'/{polymer_length}/directions.npy')
    sin_cos = np.load('../data/random/' + data_type + f'/{polymer_length}/sin_cos.npy')
    lengthOfPolymer = np.shape(directions)[1]

    # sns.histplot(sin_cos.reshape(-1,1))
    # plt.show()

    # load and reshape my dataset
    # encoding_way = opt.encoding_way

    batch_size = opt.batch_size

    # assert encoding_way == 'directions' or 'sin_cos'
    #
    # if encoding_way == 'directions':
    #     polymer_dim = opt.polymer_dim - 2
    #     input_directions = torch.Tensor(directions)
    #     my_dataset = tensor_dataset(input_directions, lengthOfPolymer, 1)

    # elif encoding_way == 'sin_cos':
    if data_type == 'on_lattices':
        polymer_dim = (opt.polymer_dim - 2) * 2
        input_sin_cos = torch.Tensor(sin_cos)
        # add some noise in the real data
        input_sin_cos += torch.randn_like(input_sin_cos) * 0.02
        my_dataset = tensor_dataset(input_sin_cos, lengthOfPolymer, 2)

    elif data_type == 'off_lattices':
        polymer_dim = (opt.polymer_dim - 1) * 2
        input_sin_cos = torch.Tensor(sin_cos)
        print(input_sin_cos.size())
        my_dataset = tensor_dataset(input_sin_cos, lengthOfPolymer, 2)

    shuffle = True
    num_worker = 0
    pin_memory = True
    my_dataloader = DataLoader(dataset=my_dataset,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               num_workers=num_worker,
                               pin_memory=pin_memory)

    # initialize models
    model = opt.model
    z_dim = opt.z_dim
    if model == 'single_node':
        gen = single_node.GeneratorNet(z_dim, polymer_dim)
        disc = single_node.DiscriminatorNet(polymer_dim)
    elif model == 'three_layers':
        gen = three_layers.GeneratorNet(z_dim, polymer_dim)
        disc = three_layers.DiscriminatorNet(polymer_dim)
    elif model == 'four_layers':
        gen = four_layers.GeneratorNet(z_dim, polymer_dim)
        disc = four_layers.DiscriminatorNet(polymer_dim)
    elif model == 'five_layers':
        gen = five_layers.GeneratorNet(z_dim, polymer_dim)
        disc = five_layers.DiscriminatorNet(polymer_dim)
    elif model == 'six_layers':
        gen = six_layers.GeneratorNet(z_dim, polymer_dim)
        disc = six_layers.DiscriminatorNet(polymer_dim)

    print(gen)
    print(disc)
    # load trained model
    load = opt.load
    print(load)
    load_path_gen = opt.load_path_gen
    load_path_disc = opt.load_path_disc
    if load:
        gen_check_point = torch.load(load_path_gen)
        gen.load_state_dict(gen_check_point['gen_state_dict'])
        disc_check_point = torch.load(load_path_disc)
        disc.load_state_dict(disc_check_point['disc_state_dict'])

    # load optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr=opt.lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=opt.lr)

    device = opt.device
    saving_step = opt.saving_step
    output_path = opt.output_models_path
    n_epochs = opt.n_epochs
    noise_type = opt.noise_type
    display_step = 100000 // batch_size
    total_epoch = opt.total_epoch

    output_path = f'../experiments/11_20/{data_type}_{polymer_length}/{model}'
    print(output_path)
    print(gen)
    loss_value_disc = {}
    loss_value_gen = {}
    for i in tqdm(range(n_epochs)):
        disc_loss, gen_loss = training_bce(gen, disc, z_dim, saving_step, my_dataloader,
                                           device, disc_opt, gen_opt, display_step, noise_type)
        # gen_loss, disc_loss = training_w_loss_gp(gen, disc, z_dim, n_epochs, my_dataloader, device, disc_opt, gen_opt,
        #                                          display_step, noise_type, 5, display=True)
        total_epoch += saving_step
        save_data.save_model(gen, disc, output_path, total_epoch)
        loss_value_disc[f'epoch{total_epoch}'] = disc_loss
        loss_value_gen[f'epoch{total_epoch}'] = gen_loss

    save_data.save_loss(loss_value_gen, loss_value_disc, output_path)

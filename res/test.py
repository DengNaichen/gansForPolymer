import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import seaborn as sns
import process_data.directions_coordinates as dc
import process_data.duplicate_and_overlap as do
import process_data.sincos as sincos
import process_data.plot_polymer_histgram as my_plot
import functions as func
import process_data.n_n_distance as nn_distance
import json
from tqdm.auto import tqdm
import fnn.four_layers as model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


num_polymer = 100000
z_dim = 8
folder_name = '11_20'
polymer_len = 32
polymer_dim = (polymer_len - 1) * 2
generator = model.GeneratorNet(z_dim, polymer_dim)
discriminator = model.DiscriminatorNet(polymer_dim)
print(generator)
a = []
epoch = []
# for i in tqdm(range(50, 2050, 50)):
i = 5000
noise = func.get_noise(num_polymer, z_dim)
check_point_gen = torch.load(
    '../experiments/' + folder_name + f'/off_lattices_{polymer_len}/four_layers_gen_{i}.pth.tar')
check_point_disc = torch.load(
    '../experiments/' + folder_name + f'/off_lattices_{polymer_len}/four_layers_disc_{i}.pth.tar')
generator.load_state_dict(check_point_gen['gen_state_dict'])
discriminator.load_state_dict(check_point_disc['disc_state_dict'])
output = generator(noise).data.numpy()
fake_output = output.reshape([num_polymer, -1, 2])



#%%
# fake_angle = sincos.sin_cos_directions_off_lattices(output)
# fake_coordinates = dc.direction_coordinate_off_lattices(fake_angle)
# nndistance = nn_distance.n_n_distance(fake_coordinates)
# # a.append(nndistance)
# # epoch.append(i)


real_sin_cos = np.load(f'../data/random/off_lattices/{polymer_len}/sin_cos.npy')
plt.hist(output.reshape(polymer_dim * num_polymer, ), alpha=.5, bins=200, label='fake')
plt.hist(real_sin_cos.reshape(polymer_dim * num_polymer, ), alpha=.5, bins=200,  label='real')
plt.legend()
plt.title(f'histogram of output(components), with epoch {i}, N = {polymer_len}')
plt.savefig('output_hist', dpi=500)
plt.show()

#
real_angle = sincos.sin_cos_directions_off_lattices(real_sin_cos)
fake_angle = sincos.sin_cos_directions_off_lattices(output)
plt.hist(real_angle.reshape(int(polymer_dim * num_polymer / 2), ), alpha=.5, bins=50, label='real')
plt.hist(fake_angle.reshape(int(polymer_dim * num_polymer / 2), ), alpha=.5, bins=50, label='fake')
plt.legend()
plt.title(f'histogram of angles, with epoch {i}, N = {polymer_len}')
plt.savefig('angle_hist', dpi=500)
plt.show()
#
# fake_coordinates = dc.direction_coordinate_off_lattices(fake_angle)
# nndistance = nn_distance.n_n_distance(fake_coordinates)
# print(nndistance ** 2)
# # plt.plot(epoch, np.array(a) / np.sqrt(7))
# # plt.plot(epoch, np.ones_like(np.array(a)), ":")
# # plt.savefig('n to n distance', dpi = 500)
# # plt.show()


sin, cos = [], []
for i in fake_output:
    for j in i:
        sin.append(j[0])
        cos.append(j[1])

# plt.hist(np.array(sin).reshape(int(num_polymer * polymer_dim / 2),), bins = 100, alpha=0.5)
# plt.show()
# plt.hist(np.array(cos).reshape(int(num_polymer * polymer_dim / 2),), bins = 100, alpha=0.5)
# plt.show()
c = np.array(sin) **2 + np.array(cos) ** 2
plt.hist(c.reshape(num_polymer * polymer_dim // 2,), bins = 100, alpha=0.5)
plt.title('histogram of $u_{x}^2 + u_{y}^2$')
plt.savefig('check normalization', dpi=500)
plt.show()

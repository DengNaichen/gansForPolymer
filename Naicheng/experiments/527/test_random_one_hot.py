import numpy as np
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import res.fnn.training as training
import res.fnn.functions as func
import os
import pandas as pd
from res.process_data.dataset import tensor_dataset
from res.fnn.generator import Generator
from res.fnn.discriminator import Discriminator
import res.process_data.process_raw_data as prd
import res.process_data.dire_and_coor as dc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#%%

# prepare the raw data
# .. mean previous layer
coordinates_input = prd.read_coordinate(16, '../../data/Coordinates.dat')
directions_input = dc.coordinates_directions_four(coordinates_input)

# plot real but random data
real = []
for i in range(100000):
    for j in range(15):
        # for k in range(2):
        real.append(directions_input[i][j][0])
# that's the real input, we assume there are 4 peaks
sns.histplot(real)
plt.title('real data, total 100,000')
plt.savefig('real.png')
plt.clf()

z_dim = 2
im_dim = 15
hidden_dim = 16
display_step = 500
lr = 0.0001
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
disc_repeats = 5
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

shuffle = True
num_worker = 0
pin_memory = True
input_tensor = torch.Tensor(directions_input)
dataset = tensor_dataset(input_tensor, 15, 1)
dataloader = DataLoader(dataset=dataset,
                        shuffle=shuffle,
                        batch_size=batch_size,
                        num_workers=num_worker,
                        pin_memory=pin_memory)


def check_output(epochs):
    output = []
    for i in range(100):
        noise = func.get_noise(16, 2)
        a = gen(noise).data.numpy()
        for j in range(16):
            for k in range(15):
                output.append(a[j][k])
    if epochs % 10 == 0:
        sns.histplot(output)
        plt.savefig(f'{epochs}_epoch.png')
        plt.clf()
    return output


def check_noise(output):
    return sum(1.1 > i > 0.9 or -.1 < i < .1 for i in output)


gen, disc, gen_opt, disc_opt = training.initialize_model(z_dim, im_dim, hidden_dim, device, lr, beta_1, beta_2)

# gen = Generator(z_dim, im_dim, hidden_dim).to(device)
# disc = Discriminator(im_dim, hidden_dim).to(device)
# gen_checkpoint = torch.load('gen_2peaks_bce.pth.tar')
# disc_checkpoint = torch.load('disc_2peaks_bce.pth.tar')
# gen.load_state_dict(gen_checkpoint['gen_state_dict'])
# disc.load_state_dict(disc_checkpoint['gen_state_dict'])

# before training, check the distribution of output
check_output(epochs=0)

total_epoch = 0
epoch_list = []
good_prediction_list = []
dir = {}

for i in range(1000):
    n_epochs = 1
    training.training_bce(gen, disc, z_dim, n_epochs, dataloader, device, disc_opt, gen_opt, display_step)
    output = check_output(total_epoch)
    good_prediction = check_noise(output)

    epoch_list.append(total_epoch)
    good_prediction_list.append(good_prediction)

    total_epoch += 1

dir['epoch'] = epoch_list
dir['good prediction'] = good_prediction_list

plt.plot(epoch_list, good_prediction_list)
plt.savefig('lalala.png')
df = pd.DataFrame(dir)
df.to_csv('output.csv')

# save the model
torch.save({'gen_state_dict': gen.state_dict()}, 'gen_2peaks_bce.pth.tar')
torch.save({'disc_state_dict': disc.state_dict()}, 'disc_2peaks_bce.pth.tar')

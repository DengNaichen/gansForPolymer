import numpy as np
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import res.fnn.training as training
import res.fnn.functions as func
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import res.process_data.process_raw_data as prd
import res.process_data.dire_and_coor as dc
import res.process_data.process_output as out
from res.process_data.dataset import tensor_dataset

random_input = np.random.randint(low=0, high=4, size=(100000,15,1)) / 4

# plot real but random data
real = []
for i in range(100000):
    for j in range(15):
        real.append(random_input[i][j][0])
# that's the real input, we assume there are 4 peaks
sns.histplot(real)
plt.savefig('real_random.png')
plt.clf()

z_dim = 4
im_dim = 15
hidden_dim = 16
display_step = 782
lr = 0.0003
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
disc_repeats = 5
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

shuffle = True
num_worker = 0
pin_memory = True
input_tensor = torch.Tensor(random_input)
dataset = tensor_dataset(input_tensor, 15, 1)
dataloader = DataLoader(dataset= dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_worker,
                            pin_memory=pin_memory)

gen, disc, gen_opt, disc_opt = training.initialize_model(z_dim, im_dim, hidden_dim,
                                                         device, lr, beta_1, beta_2)

# before training, we can the distribution of output
output = []
for i in range(100):
    noise = func.get_noise(16, 4)
    a = gen(noise).data.numpy()
    for j in range(16):
        for k in range(15):
            output.append(a[j][k])
sns.histplot(output)
plt.savefig('0_epoch.png')
plt.clf()

# training 10 epochs
n_epochs = 150
training.training_bce(gen, disc, z_dim,n_epochs,dataloader,device,disc_opt,gen_opt,display_step)

output_10_epochs = []
for i in range(100):
    noise = func.get_noise(16, 4)
    a = gen(noise).data.numpy()
    for j in range(16):
        for k in range(15):
            output_10_epochs.append(a[j][k])
sns.histplot(output_10_epochs)
plt.savefig('150_epoch.png')
plt.clf()

# 20 epochs
n_epochs += 50
training.training_bce(gen, disc, z_dim,n_epochs,dataloader,device,disc_opt,gen_opt,display_step)
output_20_epochs = []
for i in range(100):
    noise = func.get_noise(16, 4)
    a = gen(noise).data.numpy()
    for j in range(16):
        for k in range(15):
            output_20_epochs.append(a[j][k])
sns.histplot(output_20_epochs)
plt.savefig('200_epoch.png')
plt.clf()

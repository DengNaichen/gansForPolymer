import numpy as np
import torch
from scipy import integrate
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import res.fnn.training as training
import res.fnn.functions as func
import os
from res.process_data.dataset import tensor_dataset
from random import shuffle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random_input = np.random.randint(low=0, high=2, size=(100000, 15, 1))
# a = np.linspace(-1, 1, 10000)
# b = np.sin(np.pi * 2 * a) ** 2
# interval = (1 + 1) / 10000
# area = interval * b
# numbers = np.round((area / np.sum(area)) * 1500000)
# filled = []
# c = 0
# for i in numbers:
#     for j in range(int(i)):
#         filled.append(a[c])
#     c += 1
# filled.remove(filled[0])
# filled.remove(filled[100])
# # sns.histplot(filled)
# shuffle(filled)
# input_array = np.array(filled)
# input_array = input_array.reshape([50000, 15, 2])

# plot real but random data
real = []
for i in range(50000):
    for j in range(15):
        for k in range(2):
            real.append(input_array[i][j][0])
# that's the real input, we assume there are 4 peaks
sns.histplot(real)
plt.title('real data, total 100,000')
plt.savefig('real_random.png')
plt.clf()

z_dim = 2
im_dim = 30
hidden_dim = 32
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
input_tensor = torch.Tensor(input_array)
dataset = tensor_dataset(input_tensor, 15, 2)
dataloader = DataLoader(dataset=dataset,
                        shuffle=shuffle,
                        batch_size=batch_size,
                        num_workers=num_worker,
                        pin_memory=pin_memory)


def check_output(epochs):
    output = []
    for i in range(100):
        noise = func.get_noise(16, 4)
        a = gen(noise).data.numpy()
        for j in range(16):
            for k in range(30):
                output.append(a[j][k])
    sns.histplot(output)
    plt.savefig(f'{epochs}_epoch.png')
    plt.clf()


gen, disc, gen_opt, disc_opt = training.initialize_model(z_dim, im_dim, hidden_dim, device, lr, beta_1, beta_2)

# before training, check the distribution of output
check_output(epochs=0)

for i in range(10):
    n_epochs = 2
    training.training_wloss(n_epochs, dataloader, device, disc_repeats, gen, gen_opt,
                            disc, disc_opt, z_dim, c_lambda, display_step)
    check_output(n_epochs * (i + 1))

import numpy as np
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import res.fnn.training as training
import res.fnn.functions as func
import os
from res.process_data.dataset import tensor_dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

random_input = np.random.randint(low=0, high=2, size=(100000, 15, 1))

# plot real but random data
real = []
for i in range(100000):
    for j in range(15):
        # for k in range(4):
        real.append(random_input[i][j][0])
# that's the real input, we assume there are 4 peaks
sns.histplot(real)
plt.title('real data, total 100,000, [0, 1/4, 2/4, and 3/4]')
plt.savefig('real_random.png')
plt.clf()

z_dim = 2
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
    sns.histplot(output)
    plt.savefig(f'{epochs}_epoch.png')
    plt.clf()


gen, disc, gen_opt, disc_opt = training.initialize_model(z_dim, im_dim, hidden_dim, device, lr, beta_1, beta_2)

# before training, check the distribution of output
check_output(epochs=0)

for i in range(100):
    n_epochs = 10
    training.training_bce(gen, disc, z_dim, n_epochs, dataloader, device, disc_opt, gen_opt, display_step)
    check_output(n_epochs * (i + 1))

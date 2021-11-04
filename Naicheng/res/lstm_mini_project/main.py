from torch.utils.data import Dataset
import torch
import numpy as np
import res.process_data.dire_and_coor as dc
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
from dataset import tensor_dataset


class MLP(torch.nn.Module):
    """
    A two hidden-layer discriminative neural network
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(14, 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(16, 32),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden0 = nn.Sequential(
            nn.Linear(32, 64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.hidden0 = nn.Sequential(
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(16, 2),
            # torch.nn.Sigmoid()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.out(z)
        return z


if __name__ == '__main__':
    train_data = np.load('data/train_data.npy')
    train_label = np.load('data/train_label.npy')

    # test_

    lengthOfPolymer = np.shape(train_data)[1]

    shuffle = True
    num_worker = 0
    pin_memory = True
    batch_size = 8000
    input_tensor = torch.Tensor(train_data)
    input_label = torch.Tensor(train_label)
    my_dataset = tensor_dataset(input_tensor, input_label, lengthOfPolymer, 1)
    my_dataloader = DataLoader(dataset=my_dataset,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               num_workers=num_worker,
                               pin_memory=pin_memory)

    net = MLP()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()

    for i in range(1000):
        for data, labels in tqdm(my_dataloader):
            cur_batch_size = len(data)
            data = data.view(cur_batch_size, -1)
            out = net(data)
            # todo: loss function
            loss = loss_func(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save({'state_dict': net.state_dict()}, 'mlp_2000.pth.tar')

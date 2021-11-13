from torch.utils.data import Dataset
import torch
import numpy as np
import res.process_data.directions_coordinates as dc
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
from dataset import tensor_dataset
from model import MLP, LSTM


# time_step = 16


if __name__ == '__main__':
    train_data = np.load('data/train_data.npy')
    train_label = np.load('data/train_label.npy')

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

    for i in tqdm(range(2000)):
        for data, labels in my_dataloader:
            cur_batch_size = len(data)
            # for lstm
            # data = data.view(cur_batch_size, 14, 1)  # reshape input to (batch, time_step, input_size)
            # for mlp
            data = data.view(cur_batch_size, -1)
            out = net(data)
            # todo: loss function
            loss = loss_func(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i%100 == 0:
            torch.save({'state_dict': net.state_dict()}, f'models/mlp_two_hidden_{i}.pth.tar')

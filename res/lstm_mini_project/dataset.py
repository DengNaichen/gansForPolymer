import numpy as np
from torch.utils.data import Dataset
import torch


class tensor_dataset(Dataset):
    def __init__(self, input_tensor, input_label, row_dim, column_dim):
        self.input_tensor = input_tensor
        self.input_label = input_label
        self.row_dim = row_dim
        self.column_dim = column_dim

    def __len__(self):
        assert len(self.input_tensor) == len(self.input_label)
        return len(self.input_tensor)

    def __getitem__(self, index: int):
        tensor_id = index
        tensor = self.input_tensor[tensor_id]
        label = self.input_label[tensor_id]
        # convert the tensor from [x,y] to [1, x, y]
        tensor = torch.reshape(tensor, [1, self.row_dim, self.column_dim])
        # all label is 1(no label)
        tensor_label = label
        return tensor, tensor_label

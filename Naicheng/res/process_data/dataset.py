from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch


# class image_dataset(Dataset):
#
#     def __init__(self, root_dir, annotation_file, transform=None):
#         """
#         root_dir is the dir of images
#         annotation_file is the file name of csv file
#         no transform for now
#         """
#         self.root_dir = root_dir
#         self.annotations = pd.read_csv(annotation_file)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         img_id = self.annotations.iloc[index, 0]
#         img = Image.open(os.path.join(self.root_dir, img_id))
#         y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
#         if self.transform is not None:
#             img = self.transform(img)
#         # the output is a tuple
#         return img, y_label


class tensor_dataset(Dataset):
    def __init__(self, input_tensor, row_dim, column_dim):
        self.input_tensor = input_tensor
        self.row_dim = row_dim
        self.column_dim = column_dim

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, index: int):
        tensor_id = index
        tensor = self.input_tensor[tensor_id]
        # convert the tensor from [x,y] to [1, x, y]
        tensor = torch.reshape(tensor, [1, self.row_dim, self.column_dim])
        # all label is 1(no label)
        tensor_label = torch.tensor(1)
        return tensor, tensor_label

import numpy as np
import torch
from model import MLP, LSTM
from torch import nn


test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')

# load model
mlp_model = MLP()
lstm_model = LSTM()

mlp_check_point = torch.load(f'mlp_2000.pth.tar')
mlp_model.load_state_dict(mlp_check_point['state_dict'])

lstm_check_point = torch.load(f'lstm_10.pth.tar')
lstm_model.load_state_dict(lstm_check_point['state_dict'])

batch = 40000
mlp_test_data = torch.Tensor(test_data).view(batch, -1)
lstm_test_data = torch.Tensor(test_data).view(batch, 14, 1)

mlp_output = mlp_model(mlp_test_data).data.numpy()
lstm_output = lstm_model(lstm_test_data).data.numpy()

mlp_predict_list = []
lstm_predict_list = []
label_list = []

for i in range(len(test_label)):
    mlp_predict_list.append(np.argmax(mlp_output[i]))
    lstm_predict_list.append(np.argmax(lstm_output[i]))
    label_list.append(np.argmax(test_label[i]))


mlp_count = 0
lstm_count = 0

for i in range(len(mlp_predict_list)):
    if label_list[i] == mlp_predict_list[i]:
        mlp_count += 1
    if label_list[i] == lstm_predict_list[i]:
        lstm_count += 1


print(f'the accuracy of MLP is { (mlp_count/len(label_list)) * 100 }%')
print(f'the accuracy of LSTM is { (lstm_count/len(label_list)) * 100 }%')

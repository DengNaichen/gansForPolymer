import numpy as np
import torch
from model import MLP, LSTM
from torch import nn
import matplotlib.pyplot as plt


test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')

# load model
mlp_model = MLP()
lstm_model = LSTM()



batch = 40000
mlp_test_data = torch.Tensor(test_data).view(batch, -1)
lstm_test_data = torch.Tensor(test_data).view(batch, 14, 1)

mlp_acc = []
lstm_acc =[]

epoch = []

for j in range(0, 2000, 100):

    mlp_predict_list = []
    # lstm_predict_list = []
    label_list = []

    mlp_check_point = torch.load(f'models/mlp_two_hidden_{j}.pth.tar')
    mlp_model.load_state_dict(mlp_check_point['state_dict'])

    # lstm_check_point = torch.load(f'models/lstm_{j}.pth.tar')
    # lstm_model.load_state_dict(lstm_check_point['state_dict'])

    mlp_output = mlp_model(mlp_test_data).data.numpy()
    # lstm_output = lstm_model(lstm_test_data).data.numpy()

    for i in range(len(test_label)):
        mlp_predict_list.append(np.argmax(mlp_output[i]))
        # lstm_predict_list.append(np.argmax(lstm_output[i]))
        label_list.append(np.argmax(test_label[i]))


    mlp_count = 0
    # lstm_count = 0

    for i in range(len(mlp_predict_list)):
        if label_list[i] == mlp_predict_list[i]:
            mlp_count += 1
        # if label_list[i] == lstm_predict_list[i]:
        #     lstm_count += 1

    epoch.append(j)
    mlp_acc.append(mlp_count/len(label_list))
    # lstm_acc.append(lstm_count / len(label_list))

plt.plot(epoch, mlp_acc, label='mlp')
# plt.plot(epoch, lstm_acc, label='lstm')
plt.legend()
plt.savefig('mlp_4_layers_acc.png')
plt.show()

print(mlp_acc)
print(lstm_acc)


# after training 2000 epochs, the acc of mlp is 91.38%
# mlp acc = [0.5, 0.891575, 0.891775, 0.89145, 0.893075, 0.893875, 0.895425, 0.893275, 0.8945, 0.894]
# lstm acc = [0.5, 0.70775, 0.751725, 0.885775, 0.906325, 0.922225, 0.936675, 0.95385, 0.97245, 0.984025]


# print(f'the accuracy of MLP is { (mlp_count/len(label_list)) * 100 }%')
# print(f'the accuracy of LSTM is { (lstm_count/len(label_list)) * 100 }%')

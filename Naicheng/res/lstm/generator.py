import torch
from torch import nn

class LSTMGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, ):
        super(LSTMGenerator, self).__init__()
        self.input_dim = 15,
        self.hidden_dim = 64,
        num_layers = 1,
        batch_first = True,


    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

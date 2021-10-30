import torch
import matplotlib.pyplot as plt
import six_four_layers as model
import Naicheng.res.process_data.process_output as pro_out
import Naicheng.res.process_data.process_raw_data as pr
import Naicheng.res.process_data.dire_and_coor as dc
import Naicheng.res.process_data.loop_overlap as lp
from Naicheng.res.process_data.dataset import tensor_dataset
from torch.utils.data import DataLoader
import seaborn as sns
import Naicheng.res.fnn.functions as func
import Naicheng.res.process_data.analysis_model as am
import numpy as np
import json
import Naicheng.res.process_data.rg2 as rg
from Naicheng.res.fnn.training import training_bce
import Naicheng.res.process_data.one_hot as oh


four_direction_input = np.load('../../data/four_directions.npy')
gen = model.GeneratorNet()
# epoch = 1000
# model_name = f'model_gan_{epoch}.pth.tar'
# check_point_gen = torch.load(model_name)
# gen.load_state_dict(check_point_gen['gen_state_dict'])

# for i in range(6):
#     epoch = (i+1) * 100
#     model_name = f'four_one_hot/model_gan_{epoch}.pth.tar'
#     check_point_gen = torch.load(model_name)
#     gen.load_state_dict(check_point_gen['gen_state_dict'])
#     a = am.process_one_hot_model({}, gen, four_direction_input, 4)
#     print(a)

epoch = 600
model_name = f'four_one_hot/model_gan_{epoch}.pth.tar'
check_point_gen = torch.load(model_name)
gen.load_state_dict(check_point_gen['gen_state_dict'])

a = gen(func.get_noise(8, 8)).data.numpy()
print(a[0])

import torch
import seaborn as sns
import sys, os
import models.source.three_layers as model
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)

import res.functions as func


gen = model.GeneratorNet()
check_point = torch.load('models/three_layers/three_layers_gan_4000.pth.tar')
gen.load_state_dict(check_point["gen_state_dict"])

noise = func.get_noise(100000, 8)
output = gen(noise).data.numpy()
output = output.reshape(1400000)

sns.histplot(output)
plt.show()

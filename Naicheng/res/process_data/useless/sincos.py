# now I have the directions information
import numpy as np


def directions_sin_cos(directions):
    # get the information of polymers
    number_polymer = np.shape(directions)[0]
    dim_polymer = np.shape(directions)[1]
    sin_cos_total = np.zeros([number_polymer, dim_polymer, 2])

    # polymer directions numbers
    backward = 0
    right_turn = 1 / 4
    forward = 1 / 2
    left_turn = 3 / 4

    # sin_cos coordinates
    back = np.array([-1, 0])
    forto = np.array([1, 0])
    right = np.array([0, 1])
    left = np.array([0, -1])

    # a dirc for convert directions to sin cos coordinates
    convert = {
        backward: back,
        forward: forto,
        right_turn: right,
        left_turn: left
    }

    for i, direction in enumerate(directions):
        for j, each_direction in enumerate(direction):
            sin_cos_total[i][j] = convert[each_direction[0]]

    return sin_cos_total


# directions = np.load('../../../data/random/16/directions.npy')
# sin_cos = directions_sin_cos(directions)
# np.save('../../../data/random/16/sin_cos.npy', sin_cos)

# directions = np.load('../../../data/self_avoid/16/directions.npy')
# sin_cos = directions_sin_cos(directions)
# np.save('../../../data/self_avoid/16/sin_cos.npy', sin_cos)

# sin_cos = np.load('../../../data/self_avoid/16/sin_cos.npy')
# print(np.shape(sin_cos))
# import matplotlib.pyplot as plt
#
# plt.hist(sin_cos.reshape(-1, 1))
# plt.show()



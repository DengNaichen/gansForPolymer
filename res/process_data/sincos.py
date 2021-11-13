# now I have the directions information
import numpy as np


def directions_sin_cos_on_lattices(directions):
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


def directions_sin_cos_off_lattices(directions):
    number_polymer = np.shape(directions)[0]
    dim_polymer = np.shape(directions)[1]
    sin_cos_total = np.zeros([number_polymer, dim_polymer, 2])

    for i, direction in enumerate(directions):
        for j, each_direction in enumerate(direction):
            sin_cos_total[i][j] = np.array([np.sin(each_direction[0]), np.cos(each_direction[0])])

    return sin_cos_total


def sin_cos_to_directions_on_lattices(output):
    num_polymer = np.shape(output)[0]
    polymer_len = np.shape(output)[1] // 2

    output = output.reshape(num_polymer, polymer_len, 2)
    # convert sin cos coordinates to turn directions
    directions = np.zeros([num_polymer, polymer_len, 1])

    convert = {
        "backward": 0,
        "right_turn": 1/4,
        "forward": 1 / 2,
        "left_turn": 3 / 4
    }

    for index, direction in enumerate(directions):
        for j, item in enumerate(output[index]):
            if item[np.argmax(np.abs(item))] < 0:
                if np.argmax(np.abs(item)) == 0:
                    # the result is [-1,0]
                    direction[j] = convert['backward']
                else:
                    # result is [0,-1]
                    direction[j] = convert['left_turn']
            elif item[np.argmax(np.abs(item))] > 0:
                if np.argmax(np.abs(item)) == 0:
                    # the result is [1,0]
                    direction[j] = convert['forward']
                else:
                    # result is [0,1]
                    direction[j] = convert['right_turn']
    return directions


# TODO
def sin_cos_directions_off_lattices(output):
    if len(np.shape(output)) == 2:
        num_polymer = np.shape(output)[0]
        polymer_len = np.shape(output)[1] // 2
        output = output.reshape(num_polymer, polymer_len, 2)
    # convert sin cos coordinates to turn directions
    elif len(np.shape(output)) == 3:
        num_polymer = np.shape(output)[0]
        polymer_len = np.shape(output)[1]
    directions = np.zeros([num_polymer, polymer_len, 1])

    for i, direction in enumerate(directions):
        for j, each_direction in enumerate(direction):
            # temp = np.arcsin(sin_cos[i][j][0] / factor)
            temp = np.arctan(output[i][j][0] / output[i][j][1])
            if output[i][j][0] > 0 and output[i][j][1] < 0:
                each_direction[0] = temp + np.pi
            elif output[i][j][0] < 0 and output[i][j][1] < 0:
                each_direction[0] = temp - np.pi
            else:
                each_direction[0] = temp
    return directions

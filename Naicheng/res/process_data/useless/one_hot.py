import numpy as np


# encode the direction to one hot vector
# two encode way, first one is convert four direction to one hot vector
def __one_hot_four(direction):
    assert np.shape(direction) == (15, 1), "single_direction is not (15, 1)"
    one_hot_matrix = np.zeros([len(direction), 4])
    for i in range(len(direction)):
        if direction[i] == 0:  # up
            one_hot_matrix[i][0] = 1
        elif direction[i] == 1 / 3:  # down
            one_hot_matrix[i][1] = 1
        elif direction[i] == 2 / 3:  # right
            one_hot_matrix[i][2] = 1
        elif direction[i] == 1:  # left
            one_hot_matrix[i][3] = 1

    return one_hot_matrix


def one_hots_four(directions):
    one_hot_matrics = np.zeros([np.shape(directions)[0], np.shape(directions)[1], 4])
    for i in range(np.shape(directions)[0]):
        one_hot_matrics[i] = __one_hot_four(directions[i])
    return one_hot_matrics


# second one is to convert three directions to one hot vector
def __one_hot_three(direction):
    assert np.shape(direction) == (15, 1), "single_direction is not (15, 1)"
    one_hot_matrix = np.zeros([len(direction), 3])
    for i in range(len(direction)):
        if direction[i] == 0:  # forward
            one_hot_matrix[i][0] = 1
        elif direction[i] == 0.5:  # left
            one_hot_matrix[i][1] = 1
        elif direction[i] == 1:  # right
            one_hot_matrix[i][2] = 1
    return one_hot_matrix


def one_hot_three(directions):
    one_hot_matrics = np.zeros([np.shape(directions)[0], np.shape(directions)[1], 3])
    for i in range(np.shape(directions)[0]):
        one_hot_matrics[i] = __one_hot_three(directions[i])
    return one_hot_matrics


def slice_one_hot(output, num_directions):
    z_dim = np.shape(output)[0]
    one_hot_dim = 15
    if num_directions == 3:
        sliced_one_hot = np.zeros([z_dim,one_hot_dim,3])
    elif num_directions == 4:
        sliced_one_hot = np.zeros([z_dim, one_hot_dim, 4])

    for i in range(z_dim):
        for j in range(one_hot_dim):
            if num_directions == 3:
                sliced_one_hot[i][j] = output[i][j:j+3]
            elif num_directions == 4:
                sliced_one_hot[i][j] = output[i][j:j + 4]
    return sliced_one_hot


def __get_max_index(one_hot):
    return np.where(one_hot == np.max(one_hot))[0][0]


def __one_hot_to_direction(one_hot, shape):
    direction = np.ones([15, 1])
    if shape[2] == 4: # todo
        for i in range(len(one_hot)):
            if __get_max_index(one_hot[i]) == 0:  # up
                direction[i] = 0
            elif __get_max_index(one_hot[i]) == 1:  # down
                direction[i] = 1 / 3
            elif __get_max_index(one_hot[i]) == 2:  # right
                direction[i] = 2 / 3
            elif __get_max_index(one_hot[i]) == 3:  # left
                direction[i] = 1
    elif shape[2] == 3:
        for i in range(len(one_hot)):
            if __get_max_index(one_hot[i]) == 0:  # forward
                direction[i] = 0
            elif __get_max_index(one_hot[i]) == 1:  # left
                direction[i] = 0.5
            if __get_max_index(one_hot[i]) == 3:  # right
                direction[i] = 1
    return direction


def one_hots_directions(one_hots):
    one_hot_shape = np.shape(one_hots)
    directions = np.zeros((one_hot_shape[0], 15, 1))
    for i in range(one_hot_shape[0]):
        directions[i] = __one_hot_to_direction(one_hots[i], one_hot_shape)
    return directions

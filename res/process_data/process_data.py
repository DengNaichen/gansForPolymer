import numpy as np


# TODO
def four_direction(coordinate):
    assert np.shape(coordinate) == (16, 2), "single coor shape is not (16, 2)"
    direction = np.ones([15, 1])
    for i in range(0, len(coordinate) - 1):
        change = coordinate[i + 1] - coordinate[i]
        if np.array_equal(change, np.asarray([0, 1])):  # up
            direction[i] = 0
        elif np.array_equal(change, np.asarray([0, -1])):  # down
            direction[i] = 1 / 3
        elif np.array_equal(change, np.asarray([1, 0])):  # right
            direction[i] = 2 / 3
        elif np.array_equal(change, np.asarray([-1, 0])):  # left
            direction[i] = 1
    return direction


def single_cartesian(single_direction):
    assert np.shape(single_direction) == (15, 1), "single direction is not (15, 1)"

    single_cartesian_2d = np.zeros([15, 2])

    for i in range(len(single_direction)):
        if single_direction[i] == 0.:  # up
            single_cartesian_2d[i][0] = 1
        elif single_direction[i] == 1 / 3:  # down
            single_cartesian_2d[i][0] = -1
        elif single_direction[i] == 2 / 3:  # right
            single_cartesian_2d[i][1] = 1
        elif single_direction[i] == 1.:  # left
            single_cartesian_2d[i][1] = -1
    return single_cartesian_2d


def single_one_hot_four(single_direction):
    assert np.shape(single_direction) == (15, 1), "single_direction is not (15, 1)"
    single_one_hot_four_matrix = np.zeros([len(single_direction), 4])
    for i in range(len(single_direction)):
        if single_direction[i] == 0:  # up
            single_one_hot_four_matrix[i][0] = 1
        elif single_direction[i] == 1 / 3:  # down
            single_one_hot_four_matrix[i][1] = 1
        elif single_direction[i] == 2 / 3:  # right
            single_one_hot_four_matrix[i][2] = 1
        elif single_direction[i] == 1:  # left
            single_one_hot_four_matrix[i][3] = 1

    return single_one_hot_four_matrix




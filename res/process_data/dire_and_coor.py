import numpy as np
from torch.functional import cartesian_prod


# four directions
def __single_coor_direction_four(single_coor):
    assert np.shape(single_coor) == (16, 2), "single coordinates shape is not (16, 2)"

    single_direction = np.ones([15, 1])

    for i in range(0, len(single_coor) - 1):
        change = single_coor[i + 1] - single_coor[i]
        if np.array_equal(change, np.asarray([0, 1])):  # up
            single_direction[i] = 0
        elif np.array_equal(change, np.asarray([0, -1])):  # down
            single_direction[i] = 1 / 3
        elif np.array_equal(change, np.asarray([1, 0])):  # right
            single_direction[i] = 2 / 3
        elif np.array_equal(change, np.asarray([-1, 0])):  # left
            single_direction[i] = 1

    return single_direction


def coor_direction_four(coor):
    direction = np.zeros([len(coor),
                          len(coor[0]) - 1,
                          1])
    for i in range(len(coor)):
        direction[i] = __single_coor_direction_four(coor[i])

    return direction


# three directions
def __next_direction_three(coordinate, prev_index, curr_index, next_index):
    assert np.shape(coordinate) == (16, 2), "single coordinate shape is not (16, 2)"

    right = np.array([1, 0])
    left = np.array([-1, 0])
    up = np.array([0, 1])
    down = np.array([0, -1])

    prev = coordinate[curr_index] - coordinate[prev_index]
    curr = coordinate[next_index] - coordinate[curr_index]

    if np.array_equal(prev, curr):  # forward
        return 0

    else:
        if np.array_equal(prev, right):  # right
            if np.array_equal(curr, down):  # down
                return 1  # right
            elif np.array_equal(curr, up):  # up
                return 0.5  # left
        elif np.array_equal(prev, left):  # left
            if np.array_equal(curr, up):  # up
                return 1  # right
            elif np.array_equal(curr, down):  # down
                return 0.5  # left
        elif np.array_equal(prev, up):  # up
            if np.array_equal(curr, right):
                return 1  # right
            elif np.array_equal(curr, left):
                return 0.5  # left
        elif np.array_equal(prev, down):  # down
            if np.array_equal(curr, left):  # left
                return 1  # right
            elif np.array_equal(curr, right):
                return 0.5  # left


def __single_coordinate_direction_three(coordinate):
    single_direction = np.zeros([15, 1])
    for i in range(1, 15):
        single_direction[i] = __next_direction_three(coordinate, i - 1, i, i + 1)
    return single_direction


def coordinate_direction_three(coordinates):
    direction = np.zeros([len(coordinates),
                          len(coordinates[0]) - 1,
                          1])

    for i in range(len(coordinates)):
        direction[i] = __single_coordinate_direction_three(coordinates[i])
    return direction


# encode the direction to one hot vector
def __single_one_hot_four(single_direction):
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


def one_hot_four(direction):
    one_hot_four_matrix = np.zeros([np.shape(direction)[0], np.shape(direction)[1], 4])
    for i in range(np.shape(direction)[0]):
        one_hot_four_matrix[i] = __single_one_hot_four(direction[i])
    return one_hot_four_matrix


# encode the direction to cartesian coordinates
def __single_cartesian(single_direction):
    """
    single_direction_list is the numpy ndarray with shape [1,15]
    return: a ndarray with encoded information
    """
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


def cartesian_2d(direction):
    cartesian_2d_matrix = np.zeros([len(direction), 15, 2])
    for i in range(len(direction)):
        cartesian_2d_matrix[i] = __single_cartesian(direction[i])
    return cartesian_2d_matrix


# decode the information
def __round_direction_four(single_direction):
    a = [round(num * 3) / 3 for num in single_direction]
    b = np.reshape(a, (1, 15))
    return b


def __single_direction_coor_four(single_direction):
    """
    :param direction: the direction information for single polymer
    :return: the coordinate for single polymer
    """
    round_direction = __round_direction_four(single_direction)
    single_coor = np.zeros([16, 2])
    single_coor[0][0], single_coor[0][1] = 0, 0
    for i in range(len(round_direction)):
        c = np.copy(single_coor[i])
        if round_direction[i] == 1:  # left
            c[0] -= 1
        elif round_direction[i] == 2 / 3:  # right
            c[0] += 1
        elif round_direction[i] == 0:  # up
            c[1] += 1
        elif round_direction[i] == 1 / 3:  # down
            c[1] -= 1
        single_coor[i + 1] = c
    return single_coor


def direction_coor_four(direction_all):
    coor = np.ones([len(direction_all), 16, 2])
    i = 0
    for direction in direction_all:
        coor[i] = __single_direction_coor_four(direction)
        i += 1
    return coor


# decode three direction to coordinates
def __single_direction_coordinate_three(direction):
    coordinate = np.zeros([16, 2])
    up = np.array([0, 1])
    down = np.array([0, -1])
    right = np.array([1, 0])
    left = np.array([-1, 0])
    # first step is up
    coordinate[0] = np.array([0, 0])
    coordinate[1] = coordinate[0] + up

    # get the prev direction in fixed coordinates
    for i in range(1, 15):
        prev = coordinate[i] - coordinate[i - 1]
        if np.array_equal(prev, up):  # up
            if direction[i][0] == 0:  # forward
                coordinate[i + 1] = up + coordinate[i]
            elif direction[i][0] == 0.5:  # left
                coordinate[i + 1] = left + coordinate[i]
            elif direction[i][0] == 1:  # right
                coordinate[i + 1] = right + coordinate[i]

        elif np.array_equal(prev, down):  # down
            if direction[i][0] == 0:  # forward
                coordinate[i + 1] = down + coordinate[i]
            elif direction[i][0] == 0.5:  # left
                coordinate[i + 1] = right + coordinate[i]
            elif direction[i][0] == 1:  # right
                coordinate[i + 1] = left + coordinate[i]

        elif np.array_equal(prev, right):  # right
            if direction[i][0] == 0:  # forward
                coordinate[i + 1] = right + coordinate[i]
            elif direction[i][0] == 0.5:  # left
                coordinate[i + 1] = up + coordinate[i]
            elif direction[i][0] == 1:  # right
                coordinate[i + 1] = down + coordinate[i]

        elif np.array_equal(prev, left):  # left
            if direction[i][0] == 0:  # forward
                coordinate[i + 1] = left + coordinate[i]
            elif direction[i][0] == 0.5:  # left
                coordinate[i + 1] = down + coordinate[i]
            elif direction[i][0] == 1:  # up
                coordinate[i + 1] = up + coordinate[i]
    return coordinate


def direction_coordinate_four(directions):
    coordinates = np.ones([len(directions), 16, 2])
    i = 0
    for direction in directions:
        coordinates[i] = __single_direction_coordinate_three(direction)
        i += 1
    return coordinates
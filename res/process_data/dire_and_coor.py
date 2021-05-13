import numpy as np
from torch.functional import cartesian_prod


# four directions #
def __coordinate_direction_four(coordinate):
    assert np.shape(coordinate) == (16, 2), "single coordinates shape is not (16, 2)"

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


def coordinates_directions_four(coordinates):
    directions = np.zeros([len(coordinates),
                          len(coordinates[0]) - 1,
                          1])
    for i in range(len(coordinates)):
        directions[i] = __coordinate_direction_four(coordinates[i])

    return directions


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


def __coordinate_direction_three(coordinate):
    direction = np.zeros([15, 1])
    for i in range(1, 15):
        direction[i] = __next_direction_three(coordinate, i - 1, i, i + 1)
    return direction


def coordinates_directions_three(coordinates):
    directions = np.zeros([len(coordinates),
                          len(coordinates[0]) - 1,
                          1])

    for i in range(len(coordinates)):
        directions[i] = __coordinate_direction_three(coordinates[i])
    return directions


# encode the direction to one hot vector
def __one_hot(direction):
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


def one_hots(directions):
    one_hot_matrics = np.zeros([np.shape(directions)[0], np.shape(directions)[1], 4])
    for i in range(np.shape(directions)[0]):
        one_hot_matrics[i] = __one_hot(directions[i])
    return one_hot_matrics


# encode the direction to cartesian coordinates
def __sincos(single_direction):
    """
    single_direction_list is the numpy ndarray with shape [1,15]
    return: a ndarray with encoded information
    """
    assert np.shape(single_direction) == (15, 1), "single direction is not (15, 1)"

    sincos = np.zeros([15, 2])

    for i in range(len(single_direction)):
        if single_direction[i] == 0.:  # up
            sincos[i][0] = 1
        elif single_direction[i] == 1 / 3:  # down
            sincos[i][0] = -1
        elif single_direction[i] == 2 / 3:  # right
            sincos[i][1] = 1
        elif single_direction[i] == 1.:  # left
            sincos[i][1] = -1
    return sincos


def sin_cos(directions):
    sincos_matrix = np.zeros([len(directions), 15, 2])
    for i in range(len(directions)):
        sincos_matrix[i] = __sincos(directions[i])
    return sincos_matrix


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
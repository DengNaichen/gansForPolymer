import numpy as np


def slicing_output(output):
    # 根据output的长度确定一个东西
    noise_dim, dirc_dim = np.shape()[0], np.shape()[1]
    # for i in range(noise_dim):


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
            # if direction[i][0] == 0:  # forward
            if direction[i][0] == 0.1:
                coordinate[i + 1] = up + coordinate[i]
            # elif direction[i][0] == 0.5:  # left
            elif direction[i][0] == 0.5:
                coordinate[i + 1] = left + coordinate[i]
            # elif direction[i][0] == 1.0:  # right
            elif direction[i][0] == 0.9:  # right
                coordinate[i + 1] = right + coordinate[i]

        elif np.array_equal(prev, down):  # down
            # if direction[i][0] == 0:  # forward
            if direction[i][0] == 0.1:
                coordinate[i + 1] = down + coordinate[i]
            elif direction[i][0] == 0.5:  # left
                coordinate[i + 1] = right + coordinate[i]
            # elif direction[i][0] == 1:  # right
            elif direction[i][0] == 0.9:
                coordinate[i + 1] = left + coordinate[i]

        elif np.array_equal(prev, right):  # right
            # if direction[i][0] == 0:  # forward
            if direction[i][0] == 0.1:
                coordinate[i + 1] = right + coordinate[i]
            elif direction[i][0] == 0.5:  # left
                coordinate[i + 1] = up + coordinate[i]
            # elif direction[i][0] == 1:  # right
            elif direction[i][0] == 0.9:
                coordinate[i + 1] = down + coordinate[i]

        elif np.array_equal(prev, left):  # left
            # if direction[i][0] == 0:  # forward
            if direction[i][0] == 0.1:
                coordinate[i + 1] = left + coordinate[i]
            elif direction[i][0] == 0.5:  # left
                coordinate[i + 1] = down + coordinate[i]
            # elif direction[i][0] == 1:  # up
            elif direction[i][0] == 0.9:
                coordinate[i + 1] = up + coordinate[i]
    return coordinate


def direction_coordinate_three(directions):
    coordinates = np.ones([len(directions), 16, 2])
    i = 0
    for direction in directions:
        coordinates[i] = __single_direction_coordinate_three(direction)
        i += 1
    return coordinates
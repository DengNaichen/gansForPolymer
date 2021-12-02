import numpy as np


def __next_direction_on_lattices(coordinate, prev_index, curr_index, next_index):
    assert np.shape(coordinate) == (16, 2), "single coordinate shape is not (16, 2)"

    right = np.array([1, 0])
    left = np.array([-1, 0])
    up = np.array([0, 1])
    down = np.array([0, -1])

    backward = 0
    right_turn = 1/4
    forward = 1/2
    left_turn = 3/4

    prev = coordinate[curr_index] - coordinate[prev_index]
    curr = coordinate[next_index] - coordinate[curr_index]

    if np.array_equal(prev, curr):  # forward
        return forward

    else:
        if np.array_equal(prev, right):  # right
            if np.array_equal(curr, down):  # down
                return right_turn  # right
            elif np.array_equal(curr, up):  # up
                return left_turn  # left
            elif np.array_equal(curr, left): # left
                return backward # backward

        elif np.array_equal(prev, left):  # left
            if np.array_equal(curr, up):  # up
                return right_turn  # right
            elif np.array_equal(curr, down):  # down
                return left_turn  # left
            elif np.array_equal(curr, right): # right
                return backward

        elif np.array_equal(prev, up):  # up
            if np.array_equal(curr, right):
                return right_turn  # right
            elif np.array_equal(curr, left):
                return left_turn  # left
            elif np.array_equal(curr, down): # down
                return backward # backward

        elif np.array_equal(prev, down):  # down
            if np.array_equal(curr, left):  # left
                return right_turn  # right
            elif np.array_equal(curr, right):
                return left_turn  # left
            elif np.array_equal(curr, up): # up
                return backward


def __coordinate_direction_on_lattices(coordinate):
    direction = np.zeros([15, 1])
    for i in range(1, 15):
        direction[i] = __next_direction_on_lattices(coordinate, i - 1, i, i + 1)
    return direction


def coordinates_directions(coordinates):
    directions = np.zeros([len(coordinates),
                          len(coordinates[0]) - 1,
                          1])

    for i in range(len(coordinates)):
        directions[i] = __coordinate_direction_on_lattices(coordinates[i])
    return directions


# decode three direction to coordinates
def __single_direction_coordinate_on_lattices(direction, length):
    coordinate = np.zeros([length, 2])
    up = np.array([0, 1])
    down = np.array([0, -1])
    right = np.array([1, 0])
    left = np.array([-1, 0])

    # first step is up
    coordinate[0] = np.array([0, 0])
    coordinate[1] = coordinate[0] + up

    backward = 0
    right_turn = 1/4
    forward = 1/2
    left_turn = 3/4

    # get the prev direction in fixed coordinates
    for i in range(1, length - 1):
        prev = coordinate[i] - coordinate[i - 1]

        if np.array_equal(prev, up):  # up

            if direction[i][0] == forward:
                coordinate[i + 1] = up + coordinate[i]

            elif direction[i][0] == left_turn:
                coordinate[i + 1] = left + coordinate[i]

            elif direction[i][0] == right_turn:
                coordinate[i + 1] = right + coordinate[i]

            elif direction[i][0] == backward:
                coordinate[i + 1] = down + coordinate[i]

        elif np.array_equal(prev, down):  # down

            if direction[i][0] == forward:  # forward
                coordinate[i + 1] = down + coordinate[i]

            elif direction[i][0] == left_turn:  # left
                coordinate[i + 1] = right + coordinate[i]

            elif direction[i][0] == right_turn:  # right
                coordinate[i + 1] = left + coordinate[i]

            elif direction[i][0] == backward:  # backward
                coordinate[i + 1] = up + coordinate[i]

        elif np.array_equal(prev, right):  # right

            if direction[i][0] == forward:  # forward
                coordinate[i + 1] = right + coordinate[i]

            elif direction[i][0] == left_turn:  # left
                coordinate[i + 1] = up + coordinate[i]

            elif direction[i][0] == right_turn:  # right
                coordinate[i + 1] = down + coordinate[i]

            elif direction[i][0] == backward:  # backward
                coordinate[i + 1] = left + coordinate[i]

        elif np.array_equal(prev, left):  # left

            if direction[i][0] == forward:  # forward
                coordinate[i + 1] = left + coordinate[i]

            elif direction[i][0] == left_turn:  # left
                coordinate[i + 1] = down + coordinate[i]

            elif direction[i][0] == right_turn:  # up
                coordinate[i + 1] = up + coordinate[i]

            elif direction[i][0] == backward:  # backward
                coordinate[i + 1] = right + coordinate[i]
    return coordinate


def direction_coordinate_on_lattices(directions):

    num_polymer = np.shape(directions)[0]
    length_polymer = np.shape(directions)[1]

    # make the first one is forward
    directions_modified = np.zeros((num_polymer, length_polymer + 1, 1))
    for i in range(len(directions)):
        directions_modified[i] = np.vstack(([0], directions[i]))

    coordinates = np.ones([num_polymer, length_polymer + 2, 2])

    for i, direction in enumerate(directions_modified):
        coordinates[i] = __single_direction_coordinate_on_lattices(direction, length_polymer + 2)
    return coordinates


def direction_coordinate_off_lattices(directions):
    num_polymer = np.shape(directions)[0]
    len_polymer =  np.shape(directions)[1] + 1
    step_size = 1
    coordinate = np.zeros([num_polymer, len_polymer, 2])
    for i, direction in enumerate(directions):
        cumulate = 0
        for j, each_direction in enumerate(direction):
            next_step = np.array(
                [np.sin(each_direction[0] + cumulate), np.cos(each_direction[0] + cumulate)]) * step_size
            coordinate[i][j + 1] = coordinate[i][j] + next_step
            cumulate = + each_direction[0]

    return coordinate

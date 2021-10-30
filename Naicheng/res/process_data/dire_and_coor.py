import numpy as np
from torch.functional import cartesian_prod


# four directions #
# the four directions here means abs coordinate, which mean the coordinate is 'fixed' on the paper
# so we can use up, down, right, left to describe the direction
# def __coordinate_direction_four(coordinate):
#     assert np.shape(coordinate) == (16, 2), "single coordinates shape is not (16, 2)"

#     direction = np.ones([15, 1])

#     for i in range(0, len(coordinate) - 1):
#         change = coordinate[i + 1] - coordinate[i]
#         if np.array_equal(change, np.asarray([0, 1])):  # up
#             direction[i] = 0
#         elif np.array_equal(change, np.asarray([0, -1])):  # down
#             direction[i] = 1 / 3
#         elif np.array_equal(change, np.asarray([1, 0])):  # right
#             direction[i] = 2 / 3
#         elif np.array_equal(change, np.asarray([-1, 0])):  # left
#             direction[i] = 1

#     return direction


# def coordinates_directions_four(coordinates):
#     directions = np.zeros([len(coordinates),
#                           len(coordinates[0]) - 1,
#                           1])
#     for i in range(len(coordinates)):
#         directions[i] = __coordinate_direction_four(coordinates[i])

#     return directions


# three directions
def __next_direction_three(coordinate, prev_index, curr_index, next_index):
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


# encode the direction to cartesian coordinates
# def __sincos(single_direction):
#     """
#     single_direction_list is the numpy ndarray with shape [1,15]
#     return: a ndarray with encoded information
#     """
#     assert np.shape(single_direction) == (15, 1), "single direction is not (15, 1)"
#
#     sincos = np.zeros([15, 2])
#
#     for i in range(len(single_direction)):
#         if single_direction[i] == 0.:  # up
#             sincos[i][0] = 1
#         elif single_direction[i] == 1 / 3:  # down
#             sincos[i][0] = -1
#         elif single_direction[i] == 2 / 3:  # right
#             sincos[i][1] = 1
#         elif single_direction[i] == 1.:  # left
#             sincos[i][1] = -1
#     return sincos


# def sin_cos(directions):
#     sincos_matrix = np.zeros([len(directions), 15, 2])
#     for i in range(len(directions)):
#         sincos_matrix[i] = __sincos(directions[i])
#     return sincos_matrix


# decode the information
# def __round_direction_four(single_direction):
#     a = [round(num * 3) / 3 for num in single_direction]
#     b = np.reshape(a, (1, 15))
#     return b
#
#
# def __single_direction_coordinates_four(direction):
#     """
#     :param direction: the direction information for single polymer
#     :return: the coordinate for single polymer
#     """
#     coordinate = np.zeros([16, 2])
#     coordinate[0][0], coordinate[0][1] = 0, 0
#     for i in range(len(direction)):
#         c = np.copy(coordinate[i])
#         if direction[i] == 1:  # left
#             c[0] -= 1
#         elif direction[i] == 2 / 3:  # right
#             c[0] += 1
#         elif direction[i] == 0:  # up
#             c[1] += 1
#         elif direction[i] == 1 / 3:  # down
#             c[1] -= 1
#         coordinate[i + 1] = c
#     return coordinate
#
#
# def directions_coordinates_four(directions):
#     coordinates = np.ones([len(directions), 16, 2])
#     i = 0
#     for direction in directions:
#         coordinates[i] = __single_direction_coordinates_four(direction)
#         i += 1
#     return coordinates


# decode three direction to coordinates
def __single_direction_coordinate_three(direction, length):
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


def direction_coordinate_three(directions, length):
    coordinates = np.ones([len(directions), length, 2])
    i = 0
    for direction in directions:
        coordinates[i] = __single_direction_coordinate_three(direction, length)
        i += 1
    return coordinates
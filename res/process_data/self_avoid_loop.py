import numpy as np


def check_fold_cross(coordinate):
    """
    check if a single polymer is folding or crossing
    """
    assert np.shape(coordinate) == (16, 2)
    folding_count = 0
    crossing_count = 0
    for i in range(0, len(coordinate)):
        temp = coordinate[i]
        if (temp == coordinate[:i]).all(axis=1).any():
            if np.array_equal(temp, coordinate[:i][-2]):
                folding_count += 1
            else:
                crossing_count += 1
    return [folding_count, crossing_count]


def count_fold_cross(coordinates):
    """
    :param coordinates: coordinates of multiple polymer, with dim(x, 16, 2)
    :return: three int, fold count, cross count, and self avoid count
    """
    folding_count = 0
    crossing_count = 0
    self_avoid = 0
    for coordinate in coordinates:
        if check_fold_cross(coordinate)[0] != 0:
            folding_count += 1
        if check_fold_cross(coordinate)[1] != 0:
            crossing_count += 1
        elif check_fold_cross(coordinate) == [0, 0]:
            self_avoid += 1
    return folding_count, crossing_count, self_avoid


# def __direction_to_str(direction):
#     st = ""
#     for i in direction:
#         st += str(int(i[0] * 2))
#     return st


# def remove_duplicated(directions):
#     """
#     :param directions: directions array, three directions
#     :return: two numpy array, each contains batch of string that represent a polymer by three directions.
#     first one is set of strings removed duplication
#     second one is set of strings of original data
#     """
#     directions_str = []
#     removed_index = []
#     for i in range(len(directions)):
#         directions_str.append(__direction_to_str(directions[i]))
#     removed_duplicated = list(set(directions_str))
#     for j in range(len(removed_duplicated)):
#         removed_index.append(directions_str.index(removed_duplicated[j]))
#     return removed_duplicated, removed_index


# def get_unique_coor(removed_index, coordinates, directions):
#     """
#     :param removed_index: contain the index of a list that removed duplication
#     :param coordinates: coordinates of generating polymers, with duplication
#     :param directions: direction of generating polymers, with duplication
#     :return: coordinates and directions of generating polymers, without duplication
#     """
#     unique_coor = np.zeros([len(removed_index), 16, 2])
#     unique_direction = np.zeros([len(removed_index), 15, 1])
#     count = 0
#     for i in removed_index:
#         unique_coor[count] = np.copy(coordinates[i])
#         unique_direction[count] = np.copy(directions[i])
#         count += 1
#     return unique_coor, unique_direction


# def arrange_self_avoid_polymer(coordinates, self_avoid_number):
#     """
#     :param coordinates: the output coordinates
#     :param self_avoid_number: int, the number of self avoid polymer
#     :return: a sub array contain only self avoid polymer, shape of each one is (16,2)
#     """
#     self_avoid_polymers = np.zeros([self_avoid_number, 16, 2])
#     i = 0
#     for coordinate in coordinates:
#         fold_and_cross = check_fold_cross(coordinate)
#         if fold_and_cross == [0, 0]:
#             self_avoid_polymers[i] = coordinate
#             i += 1
#     return self_avoid_polymers


# def check_overlap(directions_three_input, direction_three_output):
#     output_remove_duplicated, _ = remove_duplicated(direction_three_output)
#     input_removed_duplicated, _ = remove_duplicated(directions_three_input)
#     repeat = 0
#     for output in output_remove_duplicated:
#         if output in input_removed_duplicated:
#             repeat += 1
#     return repeat

# def loop_length(coordinate):
#     """
#     :param coordinate: coordinates for single polymer
#     :return: a list contain loop length
#     """
#     assert np.shape(coordinate) == (16, 2)
#     loop = []
#     for i in range(0, len(coordinate)):
#         temp = np.copy(coordinate[i])
#         for j in range(i+1, len(coordinate)):
#             if np.array_equal(temp, coordinate[j]):
#                 loop.append(j - i)
#     return loop

from tarfile import REGTYPE
import numpy as np
import matplotlib.pyplot as plt


# convert direction to coordinates
def four_direction_to_coordinates(single_direction):
    # convert a single polymer's direction to coordinates
    assert np.shape(single_direction) == (15, 1), "single_direction is not (15, 1)"
    coordinates = np.zeros([16, 2])
    coordinates[0][0], coordinates[0][1] = 0, 0
    for i in range(len(single_direction)):
        c = np.copy(coordinates[i])
        if single_direction[i][0] == 1:  # left
            c[0] -= 1
        if single_direction[i][0] == 2/3:  # right
            c[0] += 1
        if single_direction[i][0] == 0:  # up
            c[1] += 1
        if single_direction[i][0] == 1/3:  # down
            c[1] -= 1
        coordinates[i + 1] = c
    return coordinates


# round direction if the output is direction, only for the output is scalar
def round_direction_four(single_direction):
    assert np.shape(single_direction) == (15, 1)

    return np.round(single_direction * 3)/3


# convert a single polymer's one hot vector to direction
def slicing_output(output, encoding_type):
    # can slice only one output
    if encoding_type == "onehot":
        assert len(output) == 60, "length error"
        one_hot_matrix = np.zeros([15, 4])
        for i in range(15):
            one_hot_matrix[i] = output[i*4: (i+1) * 4]
        return one_hot_matrix

    elif encoding_type == "cartesian":
        assert len(output) == 30, "length error"
        cartesian_matrix = np.zeros([15, 2])
        for i in range(15):
            cartesian_matrix[i] = output[i*2, (i+1) * 4]
        return cartesian_matrix

    elif encoding_type == "direction":
        assert len(output) == 15, "length error"
        direction = np.zeros([15, 1])
        for i in range(15):
            direction[i] = output[i]
        return direction

    else:
        print("the encoding type should be onehot, direction, or cartesian")


def __get_max_index(single_one_hot):
    assert len(single_one_hot) == 4
    return np.where(single_one_hot == np.max(single_one_hot))[0][0]


def one_hot_to_direction_four(one_hot_matrix):
    assert np.shape(one_hot_matrix) == (15, 4)

    single_direction = np.ones([15, 1])
    for i in range(len(one_hot_matrix)):
        if __get_max_index(one_hot_matrix[i]) == 0: # up
            single_direction[i] = 0
        elif __get_max_index(one_hot_matrix[i]) == 1: # down
            single_direction[i] = 1/3
        elif __get_max_index(one_hot_matrix[i]) == 2: # right
            single_direction[i] = 2/3
        elif __get_max_index(one_hot_matrix[i]) == 3: # left
            single_direction[i] = 1
    return single_direction


# convert cartesian direction
def cartesian_to_direction_four(cartesian):
    assert np.shape(cartesian) == (15, 2)

    single_direction = np.ones([15, 1])
    for i in range(len(cartesian)):
        if np.array_equal(cartesian[i], np.array([1, 0])):  # up
            single_direction[i] = 0
        elif np.array_equal(cartesian[i], np.array([-1, 0])):  # down
            single_direction[i] = 1 / 3
        elif np.array_equal(cartesian[i], np.array([0, 1])):  # right
            single_direction[i] = 2 / 3
        elif np.array_equal(cartesian[i], np.array([0, -1])):  # left
            single_direction[i] = 1
    return single_direction


def check_fold_cross(coordinate):
    """
    check if a single polymer is folding or crossing
    """
    assert np.shape(coordinate) == (16, 2)
    folding_count = 0
    crossing_count = 0
    for i in range(1, len(coordinate)):
        temp = coordinate[i]
        if (temp == coordinate[:i]).all(axis=1).any():
            if np.array_equal(temp, coordinate[:i][-2]):
                folding_count += 1
            else:
                crossing_count += 1
    return [folding_count, crossing_count]


def count_fold_cross(coordinates):
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


# get the n to n distance
def n_n_distance(coordinate):
    """
    calculate the n2n distance for single polymer
    """
    distance = np.sqrt(np.sum(np.square(np.asarray(coordinate[-1]) - np.asarray(coordinate[0]))))
    return distance


# get the statistics properties of n to n distance
def statistics(n2n_distance_array):
    """
    calculate some
    """
    mean = np.mean(n2n_distance_array)
    std = np.std(n2n_distance_array)
    mse = np.mean((n2n_distance_array - mean) ** 2)
    return mean, std, mse


# plot a polymer
def plot_polymer(coordinate):
    fig, ax = plt.subplots()
    for i in range(len(coordinate) - 1):
        ax.plot([coordinate[i][0], coordinate[i + 1][0]],
                [coordinate[i][1], coordinate[i + 1][1]],
                linewidth=5, c="black")
    plt.show()


# only self avoid polymer need to check the duplicate
def arrange_self_avoid_polymer(coordinates, self_avoid_number):
    """
    :param coordinates: the output coordinates
    :param self_avoid_number: int, the number of self avoid polymer
    :return: a sub array contain only self avoid polymer, shape of each one is (16,2)
    """
    self_avoid_polymers = np.zeros([self_avoid_number, 16, 2])
    i = 0
    for coordinate in coordinates:
        fold_and_cross = check_fold_cross(coordinate)
        if fold_and_cross == [0, 0]:
            self_avoid_polymers[i] = coordinate
            i += 1
    return self_avoid_polymers


# find the duplicate items
def __direction_to_str(direction):
    st = ""
    for i in direction:
        st += str(int(i[0] * 2))
    return st


# store all string represent polymer into an array
def remove_duplicated(directions):
    """
    :param directions: directions array, each elements
    :return: two numpy array, each contains batch of string that represent a polymer by three directions.
    """
    directions_str = []
    for i in range(len(directions)):
        directions_str.append(__direction_to_str(directions[i]))
    removed_duplicated = list(set(directions_str))
    removed_duplicated = np.array(removed_duplicated)
    directions_str = np.array(directions_str)
    return removed_duplicated, directions_str



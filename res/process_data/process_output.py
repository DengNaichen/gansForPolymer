from tarfile import REGTYPE
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig

from res.fnn.generator import Generator
from res.fnn.discriminator import Discriminator

import res.fnn.functions as func
import res.process_data.dire_and_coor as dc


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
        if single_direction[i][0] == 2 / 3:  # right
            c[0] += 1
        if single_direction[i][0] == 0:  # up
            c[1] += 1
        if single_direction[i][0] == 1 / 3:  # down
            c[1] -= 1
        coordinates[i + 1] = c
    return coordinates


# round direction if the output is direction, only for the output is scalar
def round_direction_four(single_direction):
    assert np.shape(single_direction) == (15, 1)

    return np.round(single_direction * 3) / 3


# convert a single polymer's one hot vector to direction
def slicing_output(output, encoding_type):
    # can slice only one output
    if encoding_type == "onehot":
        assert len(output) == 60, "length error"
        one_hot_matrix = np.zeros([15, 4])
        for i in range(15):
            one_hot_matrix[i] = output[i * 4: (i + 1) * 4]
        return one_hot_matrix

    elif encoding_type == "cartesian":
        assert len(output) == 30, "length error"
        cartesian_matrix = np.zeros([15, 2])
        for i in range(15):
            cartesian_matrix[i] = output[i * 2, (i + 1) * 4]
        return cartesian_matrix

    elif encoding_type == "scalar":
        assert len(output) == 15, "length error"
        direction = np.zeros([15, 1])
        for i in range(15):
            direction[i] = output[i]
        return direction

    else:
        print("the encoding type should be onehot, scalar, or cartesian")


def __get_max_index(single_one_hot):
    assert len(single_one_hot) == 4
    return np.where(single_one_hot == np.max(single_one_hot))[0][0]


def one_hot_to_direction_four(one_hot_matrix):
    assert np.shape(one_hot_matrix) == (15, 4)

    single_direction = np.ones([15, 1])
    for i in range(len(one_hot_matrix)):
        if __get_max_index(one_hot_matrix[i]) == 0:  # up
            single_direction[i] = 0
        elif __get_max_index(one_hot_matrix[i]) == 1:  # down
            single_direction[i] = 1 / 3
        elif __get_max_index(one_hot_matrix[i]) == 2:  # right
            single_direction[i] = 2 / 3
        elif __get_max_index(one_hot_matrix[i]) == 3:  # left
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


# find the duplicate items #
def __direction_to_str(direction):
    st = ""
    for i in direction:
        st += str(int(i[0] * 2))
    return st


# store all string represent polymer into an array
def remove_duplicated(directions):
    """
    :param directions: directions array, three direction
    :return: two numpy array, each contains batch of string that represent a polymer by three directions.
    """
    directions_str = []
    for i in range(len(directions)):
        directions_str.append(__direction_to_str(directions[i]))
    removed_duplicated = list(set(directions_str))
    removed_duplicated = np.array(removed_duplicated)
    directions_str = np.array(directions_str)
    return removed_duplicated, directions_str


def name_file(name, epoch):
    name = {'gen_name': name + f'_gan_{epoch}' + '.pth.tar',
            'disc_name': name + f'_disc_{epoch}' + '.pth.tar',
            'csv_name': name + f'_{epoch}' + '.csv',
            'hist_name': name + '_distance_' + f'{epoch}' + '.png',
            'output_name': name + '_output_' + f'{epoch}' + '.png'}
    return name


def save_model(gen, disc, name, epoch):

    name_dic = name_file(name, epoch)
    torch.save({'gen_state_dict': gen.state_dict()}, name_dic['gen_name'])
    torch.save({'disc_state_dict': disc.state_dict()}, name_dic['disc_name'])


def load_model(name_dic, z_dim, im_dim, hidden_dim):

    gen = Generator(z_dim, im_dim, hidden_dim).to('cpu')
    disc = Discriminator(im_dim, hidden_dim).to('cpu')

    gen_check_point = torch.load(name_dic['gen_name'])
    disc_check_point = torch.load(name_dic['disc_name'])

    gen.load_state_dict(gen_check_point['gen_state_dict'])
    disc.load_state_dict(disc_check_point['disc_state_dict'])

    return gen, disc


def get_output_coordinate(gen_model, encoding, z_dim, iteration=1000, noise_num=16):
    coordinates = np.zeros([iteration * noise_num, 16, 2])
    output_list = []
    for i in range(iteration):
        noise = func.get_noise(noise_num, z_dim)
        output = gen_model(noise).data.numpy()
        for j in range(noise_num):
            if encoding == "onehot":
                one_hot_matrix = slicing_output(output[j], "onehot")
                direction = one_hot_to_direction_four(one_hot_matrix)
                coordinate = four_direction_to_coordinates(direction)
                coordinates[(i * noise_num) + j] = coordinate

            elif encoding == "scalar":
                direction = slicing_output(output[i], 'scalar')
                direction_round = round_direction_four(direction)
                coordinate = four_direction_to_coordinates(direction_round)
                coordinates[(i * noise_num) + j] = coordinate
            for k in range(15):
                output_list.append(output[j][k])

    return coordinates, output_list


def check_overlap(directions_three_input, direction_three_output):
    output_remove_duplicated, _ = remove_duplicated(direction_three_output)
    input_removed_duplicated, _ = remove_duplicated(directions_three_input)
    repeat = 0
    for output in output_remove_duplicated:
        if output in input_removed_duplicated:
            repeat += 1
    return repeat


def check_models(name, epoch, z_dim, im_dim, hidden_dim, encoding, coordinates_input):
    # get file name
    name_dic = name_file(name, epoch)

    # load models
    gen, _ = load_model(name_dic, z_dim, im_dim, hidden_dim)

    # get output coordinates
    coordinates_output, output_list = get_output_coordinate(gen, encoding, z_dim, iteration=1000, noise_num=16)

    # check features
    folding_count, cross_count, self_avoiding = count_fold_cross(coordinates_output)

    # get sub-array contain self_avoiding, return an array(coordinates)
    self_avoiding_polymers = arrange_self_avoid_polymer(coordinates_output, self_avoiding)

    # convert input and output to three direction format
    directions_three_output = dc.coordinate_direction_three(self_avoiding_polymers)
    directions_three_input = dc.coordinate_direction_three(coordinates_input)

    # remove duplicated items from output
    removed_duplicated, directions_str = remove_duplicated(directions_three_output)

    # check overlap with raw data
    repeat = check_overlap(directions_three_input, directions_three_output)

    # get n to n distance
    distance_array = np.zeros(len(coordinates_output))
    for i in range(len(distance_array)):
        distance_array[i] = n_n_distance(coordinates_output[i])
    mean, std, mse = statistics(distance_array)

    data = {'total': [len(coordinates_output)],
            'back folding': [folding_count],
            'crossing': [cross_count],
            'self-avoid': [self_avoiding],
            'folding percentage': [folding_count / len(coordinates_output)],
            'crossing percentage': [cross_count / len(coordinates_output)],
            'self avoid percentage': [self_avoiding / len(coordinates_output)],
            'unique': [len(removed_duplicated)],
            "repeat": [repeat],
            'mean': mean,
            'std': std,
            'mse': mse}

    df = pd.DataFrame(data=data)

    # save csv
    df.to_csv(name_dic['csv_name'])

    # plot the raw output
    output_plot = sns.histplot(data=output_list, kde=True)
    plt.savefig(name_dic['output_name'], dp1=1000)
    plt.clf()

    distance_plot = sns.histplot(data=distance_array, kde=True)
    plt.savefig(name_dic['hist_name'], dp1=1000)

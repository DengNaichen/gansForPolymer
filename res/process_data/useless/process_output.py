import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


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


def get_output_coordinate(gen, encoding, z_dim, iteration=1000, noise_num=16):
    coordinates = np.zeros([iteration * noise_num, 16, 2])
    output_list = []
    for i in range(iteration):

        noise = func.get_noise(noise_num, z_dim)
        output = gen(noise).data.numpy()
        for j in range(noise_num):

            if encoding == "one-hot":
                one_hot_matrix = slicing_output(output[j], "one-hot")
                directions = one_hot_to_direction_four(one_hot_matrix)
                coordinate = four_direction_to_coordinates(directions)
                coordinates[(i * noise_num) + j] = coordinate

            elif encoding == "scalar":
                directions = slicing_output(output[i], 'scalar')
                directions = round_direction_four(directions)
                coordinate = four_direction_to_coordinates(directions)
                coordinates[(i * noise_num) + j] = coordinate

            elif encoding == 'sincos':
                sin_cos_matrix = slicing_output(output[j], 'sincos')
                directions = cartesian_to_direction_four(sin_cos_matrix)
                coordinate = four_direction_to_coordinates(directions)
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
    directions_three_output = dc.coordinates_directions_three(self_avoiding_polymers)
    directions_three_input = dc.coordinates_directions_three(coordinates_input)

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

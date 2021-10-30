import torch
import numpy as np

import process_output as pro_out
import loop_overlap as lp
from res.process_data.useless import one_hot as oh, slice_output as so
import rg2 as rg
import dire_and_coor as dc

def count_target_output(output_list, evaluate):
    count_0, count_1, count_2 = 0, 0, 0
    for i in output_list:
        if i <= 0.1:
            count_0 += 1
        elif 0.45 < i < 0.55:
            count_1 += 1
        elif i >= 0.9:
            count_2 += 1
    evaluate['count_0'] = count_0
    evaluate['count_0.5'] = count_1
    evaluate['count_1'] = count_2
    return evaluate


def check_unique(output_directions, evaluate):
    # don't need to get the unique polymers
    # but still need to check mode collapse
    remove_str, remove_index = lp.remove_duplicated(output_directions)
    # unique_coor, unique_direction = lp.get_unique_coor(remove_index, output_coordinates, output_direction)
    without_duplicate = len(remove_str)
    evaluate['unique'] = without_duplicate
    return evaluate


def check_overlap(input_three_directions, output_directions, output_coordinates, evaluate):
    overlap = pro_out.check_overlap(input_three_directions, output_directions)
    folding, crossing, self_avoid = pro_out.count_fold_cross(output_coordinates)
    self_avoid_polymers = lp.arrange_self_avoid_polymer(output_coordinates, self_avoid)
    evaluate['overlap'] = overlap
    evaluate['crossing'] = crossing
    evaluate['self_avoid'] = self_avoid
    return evaluate, self_avoid_polymers


def get_length_of_loop(output_coordinates, evaluate):
    # get the length of loop
    loop_all = []
    for i in range(len(output_coordinates)):
        if lp.check_fold_cross(output_coordinates[i]):
            loop_all += lp.loop_length(output_coordinates[i])
    # count loop length
    a = {}

    for i in loop_all:
        if loop_all.count(i) > 1:
            a[i] = loop_all.count(i)

    loop_unique = list(set(loop_all))
    for i in loop_unique:
        evaluate[f'loop{i}'] = a[i]
    return evaluate


def self_avoid_rgs(self_avoid_polymers, evaluate):
    rg2, rg4, rg6 = rg.rg2s(self_avoid_polymers)
    evaluate['rg2'] = rg2
    evaluate['rg4'] = rg4
    evaluate['rg6'] = rg6
    return evaluate


def gen_noise(size):
    n = torch.tensor(np.random.normal(0, 1, (size, 8)), dtype=torch.float32)
    return n


def process_scalar_model(evaluate, gen, input_three_directions, encode=15):
    # check the output, and get the output coordinates
    output_list = []
    num_ite = 2000
    output_directions = np.zeros([num_ite * 8, 15, 1])
    for i in range(num_ite):

        # get output
        noise = gen_noise(8)
        output = gen(noise).data.numpy()

        for j in range(8):
            for k in range(encode):
                output_list.append(output[j][k])
        # slice the output to a (n ,15 ,1)
        output_directions[8 * i: 8 * (i + 1)] = so.slice_direction(output)
    output_coordinates = dc.direction_coordinate_three(output_directions)

    evaluate = count_target_output(output_list, evaluate)
    evaluate = check_unique(output_directions, evaluate)
    evaluate, self_avoid_polymers = check_overlap(input_three_directions,
                                                  output_directions, output_coordinates, evaluate)
    evaluate = get_length_of_loop(output_coordinates, evaluate)
    evaluate = self_avoid_rgs(self_avoid_polymers, evaluate)

    return evaluate


def process_one_hot_model(evaluate, gen, input_three_directions, num_directions):
    num_ite = 2000
    # output_directions = np.zeros([num_ite * 8, 15, 1])
    assert num_directions == 3 or num_directions == 4, "num_directions should be 3 or 4"
    if num_directions == 3:
        output_one_hot = np.zeros([num_ite * 8, 15, 3])
    elif num_directions == 4:
        output_one_hot = np.zeros([num_ite * 8, 15, 4])

    for i in range(num_ite):
        # get output
        noise = gen_noise(8)
        output = gen(noise).data.numpy()

        # slice the output to a (n ,15 ,3)
        output_one_hot[8 * i: 8 * (i + 1)] = oh.slice_one_hot (output, num_directions)

    output_directions = oh.one_hots_directions(output_one_hot)
    if num_directions == 3:
        output_coordinates = dc.direction_coordinate_three(output_directions)
    elif num_directions == 4:
        output_coordinates = dc.directions_coordinates_four(output_directions)

    # don't need to consider the dimension of directions
    evaluate = check_unique(output_directions, evaluate)
    evaluate, self_avoid_polymers = check_overlap(input_three_directions,
                                                  output_directions, output_coordinates, evaluate)
    # evaluate = get_length_of_loop(output_coordinates, evaluate)
    evaluate = self_avoid_rgs(self_avoid_polymers, evaluate)

    return evaluate

import numpy as np


def n_n_distance(coordinates):
    """
    calculate the n2n distance for all polymers
    """
    summation = 0
    polymer_num = np.shape(coordinates)[0]
    for coordinate in coordinates:
        diff = coordinate[-1] - coordinate[0]
        a = diff[-1] ** 2 + diff[0]**2
        summation += a
    average = summation / polymer_num
    result = np.sqrt(average)
    return result


def rg(coordinate, polymer_len):

    x, y  = np.zeros(polymer_len), np.zeros(polymer_len)
    for i, componets in enumerate(coordinate):
        x[i] = componets[0]
        y[i] = componets[1]
    
    rg_2 = np.mean((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2)
    rg_4 = np.mean(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2) ** 4)
    rg_6 = np.mean(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2) ** 6)

    return rg_2, rg_4, rg_6


def rgs(coordinates):

    num_polymer = len(coordinates)
    len_polymer = np.shape(coordinates)[1]

    rg2_list, rg4_list, rg6_list = np.zeros(num_polymer), np.zeros(num_polymer), np.zeros(num_polymer)

    for i, coordinate in enumerate(coordinates):
        rg2_list[i], rg4_list[i], rg6_list[i] = rg(coordinate, len_polymer)
    
    return np.mean(rg2_list), np.mean(rg4_list), np.mean(rg6_list)

import numpy as np


def rg2(coordinate):
    """
    :param coordinate: coordinates of single polymer with length 16
    :return:
    """
    x = np.zeros(16)
    y = np.zeros(16)
    for i in range(16):
        x[i] = coordinate[i][0]
        y[i] = coordinate[i][1]

    rg_2 = np.mean((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2)
    rg_4 = np.mean(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2) ** 4)
    rg_6 = np.mean(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2) ** 6)
    return rg_2, rg_4, rg_6


def rg2s(coordinates):
    dim = len(coordinates)
    rg2_list, rg4_list, rg6_list = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    i = 0
    for coordinate in coordinates:
        a, b, c = rg2(coordinate)
        rg2_list[i] = a
        rg4_list[i] = b
        rg6_list[i] = c
        i += 1
    return np.mean(rg2_list), np.mean(rg4_list), np.mean(rg6_list)

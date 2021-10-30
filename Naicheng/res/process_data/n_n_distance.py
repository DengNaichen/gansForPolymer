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


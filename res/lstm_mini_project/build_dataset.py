from torch.utils.data import Dataset
import torch
import numpy as np
import res.process_data.directions_coordinates as dc
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm


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


if __name__ == '__main__':
    # get self avoid data (numpy array)
    self_avoid_directions = np.load('../../data/self_avoid/16/directions.npy')
    self_avoid_coordinates = np.load('../../data/self_avoid/16/coorinates.npy')

    # get random polymer
    random_directions = np.random.randint(0, 4, size=(120000, 14, 1)) / 4
    a = np.zeros((120000, 15, 1))
    # make the first one is forward
    for i in range(len(random_directions)):
        a[i] = np.vstack(([0], random_directions[i]))

    random_coordinates = dc.direction_coordinate_three(a, 16)

    # remove self avoid polymers
    no_self_avoid_index = []
    for i in range(len(random_coordinates)):
        if check_fold_cross(random_coordinates[i]) != [0, 0]:
            no_self_avoid_index.append(i)

    count = 0
    output = np.zeros((len(no_self_avoid_index), 14, 1))
    for i in no_self_avoid_index:
        output[count] = random_directions[i]
        count += 1

    # put two kinds of polymer together
    train_data = np.concatenate([self_avoid_directions[:80000], output[:80000]], axis=0)
    test_data = np.concatenate([self_avoid_directions[80000:], output[80000:100000]], axis=0)

    np.save('data/train_data.npy', train_data)
    np.save('data/test_data.npy', test_data)

    # # from [0:100,000], we have the label [0,1]
    # # from [100,000:-1], we have the label [1,0]
    train_label = np.zeros([len(train_data), 2])
    for i in range(len(train_data)):
        if i < 80000:
            train_label[i] = np.array([0, 1])
        else:
            train_label[i] = np.array([1, 0])

    test_label = np.zeros([len(test_data), 2])
    for i in range(len(test_label)):
        if i < 20000:
            test_label[i] = np.array([0, 1])
        else:
            test_label[i] = np.array([1, 0])

    np.save('data/train_label.npy', train_label)
    np.save('data/test_label.npy', test_label)



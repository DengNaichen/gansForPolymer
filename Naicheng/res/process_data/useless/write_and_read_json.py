import json
import numpy as np
import matplotlib.pyplot as plt
import Naicheng.res.process_data.rg2 as rg


def writ_json(data):
    """
    :param data: where the data is a python dict
    :return:
    """
    with open('data.json', 'w') as fp:
        json.dump(data, fp)


def read_json(file_path):
    """
    :param file_path: the path of json file
    :return: a python dict
    """
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    return data


def plot_tendency_with_epoch(data, epoch_start, epoch_end, step, scalar, input_coordinates):
    """
    :param scalar: boolean
    :param data: python dict
    :param epoch_start:
    :param epoch_end:
    :return:
    """
    if scalar:
        count_0 = []
        count_1 = []
        count_2 = []
    unique = []
    cross = []
    self_avoid = []
    overlap = []
    loop_4, loop_6, loop_8, loop_10, loop_12, loop_14 = [], [], [], [], [], []
    rg2, rg4, rg6 = [], [], []
    for i in range(epoch_start, epoch_end, step):
        epoch = (i + 1) * 100
        key_epoch = f'epoch{epoch}'
        if scalar:
            count_0.append(data[key_epoch]['count_0'])
            count_1.append(data[key_epoch]['count_0.5'])
            count_2.append(data[key_epoch]['count_1'])
        unique.append(data[key_epoch]['unique'])
        cross.append(data[key_epoch]['crossing'])
        self_avoid.append(data[key_epoch]['self_avoid'])
        overlap.append(data[key_epoch]['overlap'])
        loop_4.append(data[key_epoch]['loop4'])
        loop_6.append(data[key_epoch]['loop6'])
        loop_8.append(data[key_epoch]['loop8'])
        loop_10.append(data[key_epoch]['loop10'])
        loop_12.append(data[key_epoch]['loop12'])
        loop_14.append(data[key_epoch]['loop14'])
        rg2.append(data[key_epoch]['rg2'])
        rg4.append(data[key_epoch]['rg4'])
        rg6.append(data[key_epoch]['rg6'])

    if scalar:
        plt.plot(np.array(count_0) / (np.array(count_0) + np.array(count_1) + np.array(count_2)), label='forward')
        plt.plot(np.array(count_1) / (np.array(count_0) + np.array(count_1) + np.array(count_2)), label='left')
        plt.plot(np.array(count_2) / (np.array(count_0) + np.array(count_1) + np.array(count_2)), label='right')
        plt.legend(loc='best')
        plt.show()
        plt.clf()

    plt.plot(np.array(unique) / 16000, label='unique')
    plt.plot(np.array(overlap) / 16000, label='overlap')
    plt.legend(loc='best')
    plt.show()
    plt.clf()

    plt.plot(np.array(cross) / 16000, label='crossing')
    plt.plot(np.array(self_avoid) / 16000, label='self avoid')
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.show()
    plt.clf()

    plt.plot(np.array(loop_4) / 16000, label='length = 4')
    plt.plot(np.array(loop_6) / 16000, label='length = 6')
    plt.plot(np.array(loop_8) / 16000, label='length = 8')
    plt.plot(np.array(loop_10) / 16000, label='length = 10')
    plt.plot(np.array(loop_12) / 16000, label='length = 12')
    plt.plot(np.array(loop_14) / 16000, label='length = 14')
    plt.legend(loc='best')
    plt.show()
    plt.clf()

    input_rg2, input_rg4, input_rg6 = rg.rg2s(input_coordinates)
    plt.plot(np.array(rg2) / input_rg2, label='rg2')
    plt.plot(np.array(rg4) / input_rg4, label='rg4')
    plt.plot(np.array(rg6) / input_rg6, label='rg6')
    plt.legend(loc='best')
    plt.show()
    plt.clf()
import numpy as np
import pandas as pd
import re


def __str_to_int(item):
    """
    :param item: item 是一个str，格式是"-1213" 类似的
    :return: a list [-1,2,13]
    # TODO: 其实这个代码是又隐患的，没有考虑到第二个坐标数字有可能大于10的情况。
    """
    c = np.zeros(2)
    j = 0
    # 因为if中的变量会在另外一个scope中，如果要用的话，最好在外面加一个全局变量
    # 这里的a是一个int，威力暂时的储存转变过来的str
    for i in range(2):
            # 处理有负号的数字
        if item[j] == '-':
            c[i] = int(item[j] + item[j + 1])
            j += 2
        # 处理没有负号的数字
        else:
            c[i] = int(item[j])
            j += 1
    return c    


def __read_dat_file(filename):
    """
    :param filename: str, filename
    :return: a list of list contain all coordinate (not only 16)
    """
    coordinate = []
    str_info = re.compile(' ')
    for line in open(filename, "r"):
        # 去除string前后的空格和没用的东西
        item = line.rstrip()
        # 如果是空行，就跳过
        if len(item) == 0:
            continue
        # 去除字符中间的空格
        # TODO: 如果不删的话，有没有办法利用这个空格来做事？
        item = str_info.sub('', item)
        # convert str to int list
        item = __str_to_int(item)
        coordinate.append(item)
    return coordinate


def read_coordinate(num, filename):
    """
    num: the length of polymer
    filename: path of file
    """
    coor = __read_dat_file(filename)
    coordinate_all = np.zeros([len(coor)//num,num,2])
    for i in range(len(coor)//num):
        for j in range(num):
            coordinate_all[i][j] = np.asarray(coor[j + (i * num)])
    return coordinate_all

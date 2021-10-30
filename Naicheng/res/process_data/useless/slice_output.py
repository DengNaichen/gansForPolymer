import numpy as np


def slice_direction(output):

    z_dim = np.shape(output)[0]
    direction_dim = np.shape(output)[1]

    directions = np.zeros([z_dim, 15, 1])
    if direction_dim == 15:
        for i in range(z_dim):
            for j in range(direction_dim):
                directions[i][j][0] = output[i][j]

    elif direction_dim == 14:
        for i in range(z_dim):
            directions[i][0] = np.array([1])
            for j in range(direction_dim):
                directions[i][j][0] = output[i][j]

    return directions


# def slice_one_hot(output):
#
#     z_dim = np.shape(output)[0]
#     direction_dim = np.shape(output)[1]
#
#     one_hot_matrix = np.zeros([z_dim, 15, 4])
#     for i in range(z_dim):
#         for j in range(15):
#             one_hot_matrix[i][j] = output[i][j * 4: (j + 1) * 4]
#     return one_hot_matrix
#
#
# def slice_sincos(output):
#     cartesian_matrix = np.zeros([15, 2])
#     for i in range(15):
#         cartesian_matrix[i] = output[i * 2, (i + 1) * 4]
#     return cartesian_matrix
#
#
# def slicing_output(output, encoding_type):
#     # can slice only one output
#     if encoding_type == "directions":
#         return slice_direction(output)
#     elif encoding_type == 'onehot':
#         return slice_one_hot(output)
#     elif encoding_type == "sincos":
#         return slice_sincos(output)
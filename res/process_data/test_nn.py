from res.wgan.functions import get_noise
import seaborn as sns


def check_output(generator, iteration, noise_num, noise_dims, output_dims):
    """
    :param generator: the generator for testing
    :param iteration: the number of iteration
    :param noise_num: how many noise number generate once
    :param noise_dims: dimension of noise dimension
    :param output_dims: dimension for single output (for scalar, is 16, for one hot vector, is 60)
    :return: a list with len = iteration * noise_number * output_dim
    """
    output_list = []
    for k in range(iteration):
        noise = get_noise(noise_num, noise_dims)
        output = generator(noise).data.numpy()
        for i in range(noise_num):
            for j in range(output_dims):
                output_list.append(output[i][j])
    return output_list

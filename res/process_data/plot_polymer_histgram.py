import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_polymer(coordinate):
    for i in coordinate:
        plt.plot(i[0], i[1])
    plt.show()


def check_output_hist(output, epoch):

    output = output.reshape(-1,1)

    sns.histplot(output)
    plt.title(f'output of generator{epoch}')
    plt.show()


def check_component_hist(sin_cos):
    """
    the shape of sin_cos should be like:
    (x, y, 2)
    where x is the number of the polymer, 
    y is the length of the polymer, such as 15, 31, 63..
    """
    x = [] # x compoment, sin(\theta)
    y = [] # y compoment, cos(\theta)
    for i in sin_cos:
        for j in i:
            x.append(j[0])
            y.append(j[1])
    sns.histplot(x)
    plt.show()
    sns.histplot(y)
    plt.show()


def check_angle_circular_hist(directions):
    fake_directions = directions.reshape(-1,1)
    N = 1000
    a = np.zeros([1000,])
    interval = 2 * np.pi / N
    for i in fake_directions:
        a[int(i // interval)] += 1

    bottom = 2
    max_height = 4

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = max_height*np.random.rand(N)
    width = (2*np.pi) / N

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, a, width=width, bottom=bottom)

    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.8)
    plt.show()
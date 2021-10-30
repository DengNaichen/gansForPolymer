import matplotlib.pyplot as plt

def plot_polymer(coordinate):
    for i in coordinate:
        plt.plot(i[0], i[1])
    plt.show()
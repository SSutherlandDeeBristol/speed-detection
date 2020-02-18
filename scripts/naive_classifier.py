import pickle as pkl
import random
import numpy as np
import matplotlib.pyplot as plt

def calc_mse(guess, values):

    total_error = 0

    for _,s in values:
        error = (guess - s)**2
        total_error += error

    return total_error / len(values)

def calc_random_mse(values):
    return sum([(random.uniform(0, 35) - s)**2 for _,s in values]) / len(values)

if __name__ == '__main__':
    dataset = pkl.load(open('../../val/dataset_val.pkl', 'rb'))

    total_error = 0

    average_speed = sum([s for _,s in dataset.values()]) / len(dataset.values())

    # xs = np.linspace(0, 40, 1000)
    # ys = [calc_mse(x, dataset.values()) for x in xs]
    # plt.plot(xs, ys)

    plt.hist([calc_random_mse(dataset.values()) for i in range(0,1000)], 100)

    plt.show()
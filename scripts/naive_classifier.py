import pickle as pkl
import random

if __name__ == '__main__':
    dataset = pkl.load(open('../../val/dataset_val.pkl', 'rb'))

    total_error = 0

    average_speed = sum([s for _,s in dataset.values()]) / len(dataset.values())

    for _,s in dataset.values():
        random_speed = average_speed
        error = (random_speed - s)**2
        total_error += error

    mse = total_error / len(dataset.values())

    print(mse)
import matplotlib.pyplot as plt
import csv
import os
from scipy import interpolate
import numpy as np

run_name = 'bs_64_lr_0.001_run_85'
csv_path = 'csv_runs/'

def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)

def get_loss(csv_file_name):
    loss = list()

    with open(os.path.join(csv_path, csv_file_name)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for time,step,value in reader:
            loss.append([float(step),float(value)])

    return np.array(loss)

if __name__ == '__main__':

    train_loss = get_loss(f'{run_name}_train_loss.csv')
    test_loss = get_loss(f'{run_name}_test_loss.csv')

    train_loss[:,0] = (train_loss[:,0] / max(train_loss[:,0])) * 15
    test_loss[:,0] = (test_loss[:,0] / max(test_loss[:,0])) * 15

    smooth_factor = 0.6

    smoothed_train_loss = train_loss.copy()
    smoothed_test_loss = test_loss.copy()

    smoothed_train_loss[:,1] = smooth(train_loss[:,1], smooth_factor)
    smoothed_test_loss[:,1] = smooth(test_loss[:,1], smooth_factor)

    blue = (14/255, 107/255, 176/255)
    red = (197/255, 42/255, 20/255)

    plt.rc('font', size=14)

    plt.figure(figsize=(12,6))
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.xlim(left=0, right=15)
    plt.ylim(bottom=0, top=5)
    plt.xticks(np.arange(0, 16, step=1))
    plt.plot(smoothed_train_loss[:,0], smoothed_train_loss[:,1], color=blue)
    plt.plot(train_loss[:,0], train_loss[:,1], color=blue, alpha=0.2)
    plt.plot(smoothed_test_loss[:,0], smoothed_test_loss[:,1], color=red)
    plt.plot(test_loss[:,0], test_loss[:,1], color=red, alpha=0.2)
    # axes = plt.gca()
    # axes.yaxis.grid()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(csv_path, f'{run_name}_loss_curve.pdf'), bbox_inches=None, transparent=True)
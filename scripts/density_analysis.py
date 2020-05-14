import os
import csv
import numpy as np
from scipy import stats
import math
import pickle as pkl

def get_data(csv_path):
    data = list()

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for density,error in reader:
            data.append([float(density),float(error)])

    return np.array(data)

if __name__ == '__main__':

    mode = 'train'

    dataset = pkl.load(open(f'../../{mode}/dataset_{mode}.pkl', 'rb'))
    density_data = pkl.load(open(f'../../{mode}/coverage_{mode}.pkl', 'rb'))

    if mode == 'test':
        csv_path = f'plots_{mode}/coverage_error.csv'

        data = get_data(csv_path)

        bin_means, bin_edges, binnumber = stats.binned_statistic(np.array(data)[...,0],
                    np.array(data)[...,1], statistic='mean', bins=50, range=(0,1))

        np.savetxt(f'plots_{mode}/coverage_error_hist.csv',
                [(float(edge), float(val) if not math.isnan(val) else 0.0) for edge, val in zip(bin_edges[1:], bin_means)],
                delimiter=',', fmt='%s')

    density_speed = list()

    for (filename, speed) in dataset.values():
        key = filename[:17]

        density = density_data[filename]

        density_speed.append((density, speed))

    bin_means, bin_edges, bin_number = stats.binned_statistic(np.array(density_speed)[...,1],
                np.array(density_speed)[...,0], statistic='mean', bins=90, range=(0,45))

    num_in_bins = [list(bin_number).count(i) for i in range(1,len(bin_edges))]

    for i, num in enumerate(num_in_bins):
        if num < 30:
            bin_means[i-1] = 0

    np.savetxt(f'plots_{mode}/density_speed_hist_(min_num).csv',
            [(float(edge), float(val) if not math.isnan(val) else 0.0) for edge, val in zip(bin_edges[1:], bin_means)], delimiter=',', fmt='%s')

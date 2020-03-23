import pickle as pkl
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.colors as mcolors
import math
import numpy as np

run_name = 'bs_64_lr_0.001_run_79'
file_name = f'../logs/{run_name}/logits/14.pkl'

if __name__ == '__main__':
    logits = pkl.load(open(file_name, 'rb'))

    preds = [p for p,_ in logits.values()]
    labels = [l for _,l in logits.values()]

    plt.figure(figsize=(20, 12))

    plt.subplot(231)
    plt.ylim(top=7000)
    plt.title('Ground truths')
    plt.ylabel('Frequency')
    plt.xlabel('Speed (m/s)')
    plt.hist(labels, 200, range=(0,50))
    plt.subplot(232)
    plt.ylim(top=7000)
    plt.title('Predictions')
    plt.ylabel('Frequency')
    plt.xlabel('Speed (m/s)')
    plt.hist(preds, 200, range=(0,50), color='green')

    labels = []
    preds = []
    l2_errors = []
    l1_errors = []

    for k, (pred, label) in logits.items():
        l2_error = abs(pred - label)**2
        l1_errors.append(abs(pred - label))
        labels.append(label)
        l2_errors.append(l2_error)
        preds.append(pred)

    bin_means, bin_edges, binnumber = stats.binned_statistic(labels,
                l2_errors, statistic='mean', bins=100)

    plt.subplot(233)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
                label='Mean of binned squared error')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('L2 Error')
    plt.title('Mean of binned L2 error')
    plt.ylim(top=150)
    plt.savefig(f'{file_name[:-4]}.png', dpi=400)

    error_as_fraction = []
    non_zero_labels = []

    for (error, label) in zip(l1_errors, labels):
        if label > 0:
            non_zero_labels.append(label)
            error_as_fraction.append(error / label)

    mean_error_as_fraction = np.mean(error_as_fraction)
    median_error_as_fraction = sorted(error_as_fraction)[len(error_as_fraction)//2]

    print(f'Mean error/ground truth: {mean_error_as_fraction}')
    print(f'Median error/ground truth: {median_error_as_fraction}')
    print(f'MSE: {np.mean(l2_errors)}')
    print(f'Mean L1 Error: {np.mean(l1_errors)}')

    plt.subplot(234)
    plt.title('Binned L1 error/label')
    plt.ylabel('Frequency')
    plt.xlabel('L1 error/label')
    plt.hist(error_as_fraction, 200, range=(0,5), color='red')

    plt.subplot(235)
    plt.title('Mean of binned L1 error/label')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('L1 error/label')
    plt.ylim(top=5)
    bin_means, bin_edges, binnumber = stats.binned_statistic(non_zero_labels,
                error_as_fraction, statistic='mean', bins=100)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='blue', lw=2,
                label='Mean of binned error/label')

    plt.subplot(236)
    plt.title('Mean of binned L1 error')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('l1 error')
    plt.ylim(top=15)
    bin_means, bin_edges, binnumber = stats.binned_statistic(labels,
                l1_errors, statistic='mean', bins=100)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='blue', lw=2,
                label='Mean of binned l1 error')

    plt.savefig(f'{run_name}.png', dpi=400, bbox_inches=None)

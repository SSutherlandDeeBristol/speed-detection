import pickle as pkl
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.colors as mcolors
from scipy.stats import norm
from scipy import optimize
import math
import numpy as np

run_name = 'bs_64_lr_0.001_run_92'
file_name = f'../logs/{run_name}/logits/14.pkl'

def x_square_fit(x, a, b, c, d):
    return a * ((x*b)**(c)) + d

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

    (mu, sigma) = norm.fit(l2_errors)

    bin_means, bin_edges, binnumber = stats.binned_statistic(labels,
                l2_errors, statistic='mean', bins=100)

    for i, x in enumerate(bin_means):
        if math.isnan(x):
            bin_means[i] = 0

    plt.subplot(233)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
                label='Mean of binned squared error')

    xspace = np.linspace(min(labels), max(labels), 100)
    p, p_cov = optimize.curve_fit(x_square_fit, xspace, bin_means)
    plt.plot(xspace, x_square_fit(xspace, p[0], p[1], p[2], p[3]), color='red')

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

    print(f'Mean error/ground truth: {mean_error_as_fraction:.3f}')
    print(f'Median error/ground truth: {median_error_as_fraction:.3f}')
    print(f'MSE: {np.mean(l2_errors):.3f}')
    print(f'Median L2 Error: {np.median(l2_errors):.3f}')
    print(f'Mean L1 Error: {np.mean(l1_errors):.3f}')
    print(f'Median L1 Error: {np.median(l1_errors):.3f}')

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

    plt.savefig(f'plots_runs/{run_name}.png', dpi=400, bbox_inches=None)

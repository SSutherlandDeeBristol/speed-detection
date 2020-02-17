import pickle as pkl
from matplotlib import pyplot as plt
from scipy import stats

file_name = '../logs/run_2/logits/14.pkl'

if __name__ == '__main__':
    logits = pkl.load(open(file_name, 'rb'))

    error_map = {}

    labels = []
    errors = []

    for k, (pred, label) in logits.items():
        error = abs(pred - label)**2
        error_map[k] = error
        if error < 100000:
            labels.append(label)
            errors.append(error)

    error_map = sorted(error_map.items(), key=lambda item: item[1])

    print(error_map)

    bin_means, bin_edges, binnumber = stats.binned_statistic(labels,
                errors, statistic='mean', bins=50)

    plt.figure()
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
                label='Mean of binned squared error')
    plt.xlabel('Ground truth speed')
    plt.ylabel('MSE')
    plt.savefig(f'{file_name[:-4]}.png', dpi=400)
    plt.show()
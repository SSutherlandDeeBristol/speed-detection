import pickle as pkl
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.colors as mcolors

file_name = '../logs/run_3/logits/13.pkl'

if __name__ == '__main__':
    logits = pkl.load(open(file_name, 'rb'))

    preds = [p for p,_ in logits.values()]
    labels = [l for _,l in logits.values()]

    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.ylim(top=7000)
    plt.title('Histogram of the speeds in the validation dataset')
    plt.ylabel('Frequency')
    plt.xlabel('Speed (m/s)')
    plt.hist(preds, 200, range=(0,50))
    plt.subplot(132)
    plt.ylim(top=7000)
    plt.title('Histogram of the speeds predicted by the network')
    plt.ylabel('Frequency')
    plt.xlabel('Speed (m/s)')
    plt.hist(labels, 200, range=(0,50), color='green')

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

    plt.subplot(133)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
                label='Mean of binned squared error')
    plt.xlabel('Ground truth speed')
    plt.ylabel('MSE')
    plt.savefig(f'{file_name[:-4]}.png', dpi=400)
    plt.show()
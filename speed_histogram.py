import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Plot a histogram of the speeds in the data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val/test',
                    required=True)

if __name__=='__main__':

    args = parser.parse_args()

    of_map = pkl.load(open(f'../optical-flow/{args.mode}/optical_flow_map_{args.mode}.pkl', 'rb'))

    plt.hist(np.concatenate([[s for (_,s) in speeds] for speeds in of_map.values()]), bins='auto')
    plt.title('Histogram of speeds in the training set.')
    plt.xlabel('Ground truth speed')
    plt.ylabel('Binned frequency')
    plt.savefig(f'{args.mode}-speed-histogram.png', dpi=400)
    plt.show()
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import argparse

# plt.rcParams['figure.dpi'] = 400

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

    of_map = pkl.load(open(f'../../{args.mode}/dataset_{args.mode}.pkl', 'rb'))

    print(of_map)
    print(max([s for _,s in of_map.values()]))
    print(len(of_map))

    dpi = 300
    plt.figure(dpi=dpi, figsize=(20, 10))
    plt.rc('font', size=32)

    color = '#80A1C1' if args.mode == 'train' else 'orange'
    # plt.title(f'Histogram of speeds in the {args.mode}ing set.', fontsize=16)
    plt.xlabel('Ground truth speed ($ms^{-1}$)', fontsize=34, labelpad=30)
    plt.ylabel('Binned frequency', fontsize=34, labelpad=30)
    plt.hist([s for _,s in of_map.values()], bins='auto', color=color)
    plt.tight_layout()
    plt.savefig(f'plots_{args.mode}/{args.mode}-speed-histogram.png', dpi=dpi, transparent=True, bbox_inches=None)
    # plt.show()
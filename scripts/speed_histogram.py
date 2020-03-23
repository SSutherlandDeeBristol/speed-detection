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
    plt.figure(dpi=dpi, figsize=(18, 6))
    # plt.title(f'Histogram of speeds in the {args.mode}ing set.', fontsize=16)
    plt.xlabel('Ground truth speed ($ms^{-1}$)', fontsize=30, labelpad=30)
    plt.xticks(fontsize=24)
    plt.ylabel('Binned frequency', fontsize=30, labelpad=30)
    plt.yticks(fontsize=24)
    plt.hist([s for _,s in of_map.values()], bins='auto', color='#80A1C1')
    plt.tight_layout()
    plt.savefig(f'{args.mode}-speed-histogram.png', dpi=dpi, transparent=True, bbox_inches=None)
    # plt.show()
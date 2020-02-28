import pickle as pkl
import argparse
import os
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Number the optical flow images and save a map for use in a DataLoader.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val/test',
                    required=True)

parser.add_argument('--size',
                    required=True)

if __name__=='__main__':
    args = parser.parse_args()

    mode = args.mode
    size = args.size

    of_path = f'../../{mode}/'

    of_map = pkl.load(open(os.path.join(of_path, f'optical_flow_map_{mode}.pkl'), 'rb'))

    data_list = sum(of_map.values(), [])

    # plt.hist([s for f,s in data_list if f.startswith('000000001')], bins='auto')
    # plt.show()
    # sys.exit()

    zero_speeds = list(filter(lambda x: x[1] <= 0.5, data_list))

    non_zero_speeds = list(filter(lambda x: x[1] > 0.5, data_list))

    random.shuffle(zero_speeds)
    random.shuffle(non_zero_speeds)

    if mode != 'val':
        zero_speeds = zero_speeds[:int(size)//50]
        non_zero_speeds = non_zero_speeds[:(int(size) * 49)//50]

    data_list = zero_speeds + non_zero_speeds

    random.shuffle(data_list)

    dataset = {i:d for i,d in enumerate(data_list)}

    pkl.dump(dataset, open(os.path.join(of_path, f'dataset_{mode}.pkl'), 'wb'))
import pickle as pkl
import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser(
    description="Number the optical flow images and save a map for use in a DataLoader.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val/test',
                    required=True)

if __name__=='__main__':
    args = parser.parse_args()

    mode = args.mode

    of_path = f'../optical-flow/{mode}/'

    of_map = pkl.load(open(os.path.join(of_path, f'optical_flow_map_{mode}.pkl'), 'rb'))

    data_list = sum(of_map.values(), [])

    random.shuffle(data_list)

    dataset = {i:d for i,d in enumerate(data_list)}

    pkl.dump(dataset, open(os.path.join(of_path, f'dataset_{mode}.pkl'), 'wb'))
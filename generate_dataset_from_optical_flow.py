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

parser.add_argument('--num-zero-speeds',
                    required=True)

if __name__=='__main__':
    args = parser.parse_args()

    mode = args.mode
    num_zero_speeds = args.num_zero_speeds

    of_path = f'../{mode}/'

    of_map = pkl.load(open(os.path.join(of_path, f'optical_flow_map_{mode}.pkl'), 'rb'))

    data_list = sum(of_map.values(), [])

    zero_speeds = list(filter(lambda x: x[1] == 0.0, data_list))

    non_zero_speeds = list(filter(lambda x: x[1] > 0.0, data_list))

    random.shuffle(zero_speeds)

    if len(zero_speeds) >= int(num_zero_speeds) and mode != 'val':
        zero_speeds = zero_speeds[:int(num_zero_speeds)]

    data_list = zero_speeds + non_zero_speeds

    random.shuffle(data_list)

    dataset = {i:d for i,d in enumerate(data_list)}

    pkl.dump(dataset, open(os.path.join(of_path, f'dataset_{mode}.pkl'), 'wb'))
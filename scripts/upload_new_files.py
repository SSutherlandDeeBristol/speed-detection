import pexpect
import pickle as pkl
import os
import argparse

parser = argparse.ArgumentParser(
    description="Upload the new files to bc4.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val',
                    required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    of_image_dir = (f'../../{args.mode}/')

    new_files = pkl.load(open(os.path.join(of_image_dir, f'new_files_{args.mode}.pkl'), 'rb'))

    for i, key in enumerate(new_files):
        if os.path.exists(os.path.join(of_image_dir, key)):
            print(f'{key} ({i+1}/{len(new_files)})')
            output = pexpect.run(f'scp -r {os.path.join(of_image_dir, key)} bc4-external:/mnt/storage/scratch/ss15060/{args.mode}/')

    dataset_dir = os.path.join(of_image_dir, f'dataset_{args.mode}.pkl')
    of_map_dir = os.path.join(of_image_dir, f'optical_flow_map_{args.mode}.pkl')

    output = pexpect.run(f'scp {dataset_dir} bc4-external:/mnt/storage/scratch/ss15060/{args.mode}/')
    print(dataset_dir)
    output = pexpect.run(f'scp {of_map_dir} bc4-external:/mnt/storage/scratch/ss15060/{args.mode}/')
    print(of_map_dir)
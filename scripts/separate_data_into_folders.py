import pickle as pkl
import os
import argparse
import shutil

parser = argparse.ArgumentParser(
    description="Separates the images into folders to help read times.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val/test',
                    required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    mode = args.mode

    path = f'../../{mode}/'

    dataset = pkl.load(open(os.path.join(path, f'dataset_{mode}.pkl'), 'rb'))
    optical_flow_map = pkl.load(open(os.path.join(path, f'optical_flow_map_{mode}.pkl'), 'rb'))

    for k,v in optical_flow_map.items():
        new_dir = os.path.join(path, k)

        try:
            os.mkdir(new_dir)
        except FileExistsError:
            print("Directory already exists..")

        for file_name, speed in v:
            # copy image into directory
            image_path = os.path.join(path, file_name)
            new_image_path = os.path.join(new_dir, file_name)

            shutil.move(image_path, new_image_path)
import numpy as np
import cv2 as cv
import pickle as pkl
import os
import argparse

parser = argparse.ArgumentParser(
    description="Generate the optical flow images for the frames collected from video.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val/test',
                    required=True)

parser.add_argument('--clean',
                    action='store_true',
                    help='remove deleted videos entries from the map.')

if __name__ == '__main__':
    args = parser.parse_args()

    mode = args.mode

    image_dir = f'../../images/{mode}/'
    vid_dir = f'../../videos/{mode}/'
    of_dir = f'../../{mode}/'

    image_map = pkl.load(open(os.path.join(image_dir, f'image_map_{mode}.pkl'), 'rb'))

    of_map_file = os.path.join(of_dir, f'optical_flow_map_{mode}.pkl')

    of_map = {} if not os.path.isfile(of_map_file) else pkl.load(open(of_map_file, 'rb'))

    new_files = []

    if args.clean:
        for k, v in of_map:
            if not os.path.isfile(os.path.join(vid_dir, f'{k}.mov')):
                print(f'Removing {k}..')
                for (prev, current, _) in v:
                    os.remove(os.path.join(of_dir, prev))
                    os.remove(os.path.join(of_dir, current))
                of_map.pop(k, None)
    else:
        num_processed = 0

        total_num_speeds = sum([len(speeds) for speeds in image_map.values()])

        for i, (k, v) in enumerate(image_map.items()):
            if k in of_map.keys():
                print(f'{k} already been processed..')
                continue

            parent_folder = os.path.join(of_dir, k)

            if k not in new_files:
                new_files.append(k)

            try:
                os.mkdir(parent_folder)
            except FileExistsError:
                pass

            for j, (prev, current, speed) in enumerate(v):
                print(f'{k}-{j} ({num_processed + 1})| {prev} -> {current} | {speed}m/s')

                prev_path = os.path.join(image_dir, k, prev)
                current_path = os.path.join(image_dir, k, current)

                prev_image = cv.imread(prev_path)
                current_image = cv.imread(current_path)

                prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
                current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

                mask = np.zeros_like(current_image)

                hsv_current = cv.cvtColor(current_image, cv.COLOR_RGB2HSV)
                mask[:,:,1] = hsv_current[:,:,1]

                flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 15, 2, 5, 1.3, 0)

                magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

                mask[..., 0] = angle * (180 / np.pi / 2)

                mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

                rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

                cv.imwrite(os.path.join(parent_folder, f'{k}-{j}.png'), rgb)

                of_map.setdefault(k, []).append((f'{k}-{j}.png', speed))

                num_processed += 1

    pkl.dump(of_map, open(os.path.join(of_dir, f'optical_flow_map_{mode}.pkl'), 'wb'))
    pkl.dump(new_files, open(os.path.join(of_dir, f'new_files_{mode}.pkl'), 'wb'))
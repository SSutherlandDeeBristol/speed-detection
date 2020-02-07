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
                    default='train'
                    help='train/val/test',
                    required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    mode = args.mode

    image_dir = f'../images/{mode}/'
    of_dir = f'../optical-flow/{mode}/'

    image_map = pkl.load(open(os.path.join(image_dir, 'image_map.pkl'), 'rb'))

    of_map_file = os.path.join(of_dir, 'optical_flow_map.pkl')

    of_map = {} if not os.path.isfile(of_map_file) else pkl.load(open(of_map_file, 'rb'))

    num_processed = 0

    for i, (k, v) in enumerate(image_map.items()):
        if k in of_map.keys():
            print(f'{k} already been processed..')
            continue

        for j, (prev, current, speed) in enumerate(v):
            print(f'{k}-{j} ({num_processed + 1})| {prev} -> {current} | {speed}m/s')

            prev_path = os.path.join(image_dir, prev)
            current_path = os.path.join(image_dir, current)

            prev_image = cv.imread(prev_path)
            current_image = cv.imread(current_path)

            prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
            current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

            mask = np.zeros_like(current_image)
            mask[..., 1] = 255

            flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 20, 5, 5, 1.1, 0)

            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

            mask[..., 0] = angle * 180 / np.pi / 2

            mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

            rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

            cv.imwrite(os.path.join(of_dir, f'{k}-{j}.jpg'), rgb)

            of_map.setdefault(k, []).append((f'{k}-{j}.jpg', speed))

            num_processed += 1

    pkl.dump(of_map, open(os.path.join(of_dir, 'optical_flow_map.pkl'), 'wb'))
import pickle as pkl
import cv2 as cv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ =='__main__':
    mode = 'train'

    of_path = f'../../{mode}/'
    video_path = f'../../speed-challenge/videos/train.mp4'
    ground_truth_path = f'../../speed-challenge/videos/train.txt'

    of_map = pkl.load(open(os.path.join(of_path, f'optical_flow_map_{mode}.pkl'), 'rb'))

    vid = cv.VideoCapture(video_path)

    try:
        os.mkdir(os.path.join(of_path, '00000000100000000'))
    except FileExistsError:
        pass

    ground_truth_speeds = []

    for ground_truth in open(ground_truth_path, 'rb'):
        ground_truth_speeds.append(float(ground_truth))

    try:
        success, prev_image = vid.read()
    except:
        print('Could not read frame.. exiting.')
        sys.exit()

    counter = 1

    of_map['00000000100000000'] = []

    while success and counter < 20400:
        success, current_image = vid.read()

        if not success:
            break

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

        prev_image = current_image.copy()

        file_name = f'00000000100000000-{counter}.png'
        file_path = os.path.join(of_path, '00000000100000000', file_name)

        cv.imwrite(file_path, rgb)

        of_map.setdefault('00000000100000000', []).append((file_name, ground_truth_speeds[counter]))

        print(f'{file_name} | {ground_truth_speeds[counter]:.2f}m/s')

        counter = counter + 1

    pkl.dump(of_map, open(os.path.join(of_path, f'optical_flow_map_{mode}.pkl'), 'wb'))
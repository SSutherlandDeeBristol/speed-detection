import os
import cv2 as cv
import numpy as np
import torch

prev_path = '../../images/train/00d4b6b7-a0b1a3e0/00d4b6b7-a0b1a3e0-4-prev.png'
current_path = '../../images/train/00d4b6b7-a0b1a3e0/00d4b6b7-a0b1a3e0-4-current.png'

if __name__ == '__main__':

    prev_image = cv.imread(prev_path)
    current_image = cv.imread(current_path)

    prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

    mask_bgr = np.zeros_like(current_image)

    hsv_current_bgr = cv.cvtColor(current_image, cv.COLOR_BGR2HSV)

    mask_bgr[:,:,1] = hsv_current_bgr[:,:,1]

    flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 15, 2, 5, 1.3, 0)

    print(flow)

    print(flow.size)

    print(flow[..., 0])

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    mask_bgr[..., 0] = angle * (180 / np.pi / 2)

    mask_bgr[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    bgr = cv.cvtColor(mask_bgr, cv.COLOR_HSV2BGR)

    cv.imwrite(os.path.join(f'test_bgr.png'), bgr)
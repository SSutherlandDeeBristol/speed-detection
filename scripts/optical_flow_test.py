import os
import cv2 as cv
import numpy as np

prev_path = '../images/train/12841-prev.jpg'
current_path = '../images/train/12841-current.jpg'

if __name__ == '__main__':

    prev_image = cv.imread(prev_path)
    current_image = cv.imread(current_path)

    prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(current_image)
    mask[..., 1] = 255

    flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 10, 10, 5, 1.1, 0)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    mask[..., 0] = angle * 180 / np.pi / 2

    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    cv.imwrite('optical_flow_test.jpg', rgb)
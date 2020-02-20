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

    hsv_current_rgb = cv.cvtColor(current_image, cv.COLOR_RGB2HSV)
    hsv_current_bgr = cv.cvtColor(current_image, cv.COLOR_BGR2HSV)

    mask_rgb[:,:,1] = hsv_current_rgb[:,:,1]
    mask_bgr[:,:,1] = hsv_current_bgr[:,:,1]

    flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 15, 2, 5, 1.3, 0)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    mask_rgb[..., 0] = angle * (180 / np.pi / 2)
    mask_bgr[..., 0] = angle * (180 / np.pi / 2)

    mask_rgb[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    mask_bgr[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    mask_rgb = np.asarray(mask, dtype=np.float32)
    mask_bgr = np.asarray(mask, dtype=np.float32)

    rgb = cv.cvtColor(mask_rgb, cv.COLOR_HSV2RGB)
    bgr = cv.cvtColor(mask_bgr, cv.COLOR_HSV2BGR)

    cv.imwrite(os.path.join(f'test_rgb.png'), rgb)
    cv.imwrite(os.path.join(f'test_bgr.png'), bgr)
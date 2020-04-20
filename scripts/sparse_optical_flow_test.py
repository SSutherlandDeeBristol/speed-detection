import os
import cv2 as cv
import numpy as np

prev_path = '../../images/train/00d4b6b7-a0b1a3e0/00d4b6b7-a0b1a3e0-4-prev.png'
current_path = '../../images/train/00d4b6b7-a0b1a3e0/00d4b6b7-a0b1a3e0-4-current.png'

def change_brightness(image):
    bright_factor = 0.2 + np.random.uniform()
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor

    # change back to RGB
    image_bgr = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return image_bgr

if __name__ == '__main__':

    prev_image = cv.imread(prev_path)
    current_image = cv.imread(current_path)

    prev_image = change_brightness(prev_image)
    current_image = change_brightness(current_image)

    cv.imwrite('current_brightness_change.png', current_image)

    prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

    mask_bgr = np.zeros_like(current_image)

    hsv_current_bgr = cv.cvtColor(current_image, cv.COLOR_BGR2HSV)

    mask_bgr[:,:,1] = hsv_current_bgr[:,:,1]

    flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 15, 2, 5, 1.3, 0)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    mask_bgr[..., 0] = angle * (180 / np.pi / 2)

    mask_bgr[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    bgr = cv.cvtColor(mask_bgr, cv.COLOR_HSV2BGR)

    cv.imwrite(os.path.join(f'test_bgr.png'), bgr)

    # # params for ShiTomasi corner detection
    # feature_params = dict( maxCorners = 100,
    #                     qualityLevel = 0.3,
    #                     minDistance = 7,
    #                     blockSize = 7 )
    # # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize  = (15,15),
    #                 maxLevel = 2,
    #                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # # Create some random colors
    # color = np.random.randint(0,255,(100,3))
    # # Take first frame and find corners in it
    # p0 = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    # # Create a mask image for drawing purposes
    # mask = np.zeros_like(current_image)

    # # calculate optical flow
    # p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
    # # Select good points
    # good_new = p1[st==1]
    # good_old = p0[st==1]
    # # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new, good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     current_image = cv.circle(current_image,(a,b),5,color[i].tolist(),-1)
    # img = cv.add(current_image, mask)
    # cv.imwrite('sparse.png', img)
import numpy as np
import cv2 as cv
import pickle as pkl

of_dir = '../optical-flow/train/'
image_dir = '../images/train/'

if __name__ == '__main__':

    image_map = pkl.load(open('image_map.pkl', 'rb'))

    of_map = {}

    for i, (k, (prev, current, speed)) in enumerate(image_map.items()):
        print(f'{i} | {prev} -> {current} | {speed}m/s')

        prev_path = os.path.join(image_dir, prev)
        current_path = os.path.join(image_dir, current)

        prev_image = cv.imread(prev_path)
        current_image = cv.imread(current_path)

        prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
        current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255

        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 20, 5, 5, 1.1, 0)

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        mask[..., 0] = angle * 180 / np.pi / 2

        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        cv.imwrite(open(os.path.join(of_dir, f'{k}.jpg'), 'wb'), rgb)

        of_map[f'{k}.jpg'] = speed

    pkl.dump(of_map, open('optical_flow_map.pkl', 'wb'))
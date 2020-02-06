import numpy as np
import cv2 as cv
import pickle as pkl

if __name__ == '__main__':

    image_map = pkl.load(open('image_map.pkl', 'rb'))

    for i, (k, (prev, current, speed)) in enumerate(image_map.items()):
        print(f'{i} | {prev} -> {current} | {speed}m/s')
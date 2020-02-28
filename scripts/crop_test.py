import cv2 as cv
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

def crop_image(image):
    crop = transforms.CenterCrop((int(image.size[1] * 0.6), int(image.size[0] * 0.9)))
    return np.array(crop(image))

def change_brightness(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor

    # change back to RGB
    image_rgb = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)
    return image_rgb

if __name__ == '__main__':
    prev_image_path = f'../../images/train/00a04f65-8c891f94/00a04f65-8c891f94-4-prev.png'
    current_image_path = f'../../images/train/00a04f65-8c891f94/00a04f65-8c891f94-4-current.png'

    prev_image = Image.open(prev_image_path)
    prev_image.load()
    current_image = Image.open(current_image_path)
    current_image.load()

    bright_factor = 0.2 + np.random.uniform()

    prev_image = crop_image(prev_image)
    current_image = crop_image(current_image)

    cv.imshow("yah", current_image)
    cv.waitKey(0)

    prev_image = change_brightness(prev_image)
    current_image = change_brightness(current_image)

    cv.imshow("yah", current_image)
    cv.waitKey(0)

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

    cv.imshow("yah", rgb)
    cv.waitKey(0)


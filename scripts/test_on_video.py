import glob
import cv2 as cv
import torch
from torchvision import transforms
import ffmpeg
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('../network')
from cnn import CNN

def get_rotation_correct(path):
    meta_dict = ffmpeg.probe(path)

    rotation = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotation = cv.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotation = cv.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotation = cv.ROTATE_90_COUNTERCLOCKWISE

    return rotation

if __name__ == '__main__':
    video_key = '0a0ceca1-4148e482'
    model_name = 'bs_64_lr_0.001_run_69'

    model_path = f'../../models/{model_name}.pt'
    video_path = f'../../speed-challenge/videos/train.mp4'
    ground_truth_path = f'../../speed-challenge/videos/train.txt'

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    line_type = 2

    model = CNN(640, 360, 3)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    resize_transform = transforms.Resize((640, 360))

    vid = cv.VideoCapture(video_path)

    # rotation = get_rotation_correct(video_path)
    rotation = None

    ground_truth_speeds = []

    for ground_truth in open(ground_truth_path, 'rb'):
        ground_truth_speeds.append(float(ground_truth))

    # print(ground_truth_speeds)

    # plt.hist(ground_truth_speeds, bins='auto')
    # plt.show()

    try:
        success, prev_image = vid.read()
    except:
        print('Could not read frame.. exiting.')
        sys.exit()

    if rotation is not None:
        prev_image = cv.rotate(prev_image, rotation)

    counter = 1

    while success:
        success, current_image = vid.read()

        if rotation is not None:
            current_image = cv.rotate(current_image, rotation)

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

        cv.imshow("rgb", rgb)
        cv.waitKey(0)

        PIL_Image = Image.fromarray(rgb)

        PIL_Image = resize_transform(PIL_Image)

        PIL_Image = np.transpose(PIL_Image, (2,1,0))

        PIL_Image = np.expand_dims(PIL_Image, axis=0)
        with torch.no_grad():
            speed = model.forward(torch.Tensor(PIL_Image))

        print(speed)

        prev_image = current_image.copy()

        cv.putText(current_image,
            f'Speed: {speed.item():.2f}',
            (300,50),
            font,
            font_scale,
            (255,0,0),
            line_type)

        cv.putText(current_image,
            f'Ground truth: {float(ground_truth_speeds[counter]):.2f}',
            (300,100),
            font,
            font_scale,
            (0,0,255),
            line_type)

        cv.imshow("yah", current_image)
        cv.waitKey(0)

        counter = counter + 1
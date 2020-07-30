import os
import numpy as np
import torch
import time
import sys
import argparse
from torchvision import transforms
from PIL import Image

sys.path.append('../network')
from cnn import CNN

prev_image_path = '../../images/train/00d4b6b7-a0b1a3e0/00d4b6b7-a0b1a3e0-4-prev.png'
current_image_path = '../../images/train/00d4b6b7-a0b1a3e0/00d4b6b7-a0b1a3e0-4-current.png'

optical_flow_image_path = '../../train/00d4b6b7-a0b1a3e0/00d4b6b7-a0b1a3e0-4.png'
run_name = 'bs_64_lr_0.001_run_85'
model_path = f'../network/logs/{run_name}/model.pt'

parser = argparse.ArgumentParser(
    description="Test timings.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode',
                    default='optical-flow',
                    help="optical-flow or forward.")

if __name__ == '__main__':

    args = parser.parse_args()

    num_iterations = 100

    if args.mode == 'optical-flow':
        import cv2 as cv

        prev_image = cv.imread(prev_image_path)
        current_image = cv.imread(current_image_path)

        prev_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
        current_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

        start_time = time.time()

        for _ in range(num_iterations):
            flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 15, 2, 5, 1.3, 0)

        end_time = time.time()

    elif args.mode == 'forward':

        model = CNN(640, 360, 3)

        model.load_state_dict(torch.load(model_path))
        model.eval()

        resize_transform = transforms.Resize((640, 360))

        PIL_Image = Image.open(optical_flow_image_path)

        PIL_Image = resize_transform(PIL_Image)

        PIL_Image = np.transpose(PIL_Image, (2,1,0))

        PIL_Image = np.expand_dims(PIL_Image, axis=0)

        tensor_image = torch.Tensor(PIL_Image)

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                speed = model.forward(tensor_image)

        end_time = time.time()

    total_time = end_time - start_time

    average_time = total_time / num_iterations

    num_per_second = (1/average_time)

    print(f'average time taken: {average_time * 1000:.2f} ms')
    print(f'number per second: {num_per_second:.2f}')
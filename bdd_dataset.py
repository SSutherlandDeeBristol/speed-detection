import torch
import pickle
import os
import cv2 as cv

class BDDDataset(data.Dataset):
    def __init__(self, of_image_path, of_map_path, transforms):
        self.of_image_path = of_image_path
        self.transforms = transforms
        self.of_map = pickle.load(open(of_map_path, 'rb'))

    def __getitem__(self, index):
        image_name, speed = self.of_map[index]

        image_path = os.path.join(self.of_image_path, image_name)

        image = cv2.imread(image_path)

        return image, speed

    def __len__(self):
        return len(self.of_map.keys())


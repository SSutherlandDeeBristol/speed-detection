import torch
import pickle
import os
from torchvision import transforms
from torch.utils import data
import numpy as np
from PIL import Image

class BDDDataset(data.Dataset):
    def __init__(self, of_path, of_map_name, transforms):
        self.of_path = of_path
        self.transforms = transforms
        self.of_map = pickle.load(open(os.path.join(of_path, of_map_name), 'rb'))

    def __getitem__(self, index):
        # Load sample
        image_name, speed = self.of_map[index]

        image_path = os.path.join(self.of_path, image_name[:17], image_name)

        # Load optical flow image
        image = Image.open(image_path)
        image.load()

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        return image, speed, image_name

    def __len__(self):
        return len(self.of_map.keys())


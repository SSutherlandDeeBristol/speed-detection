import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, image_width, image_height, input_channels):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.input_channels = input_channels

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=24,
            kernel_size=(5, 5),
            padding=(2, 2),
            stride=(2, 2),
            bias=False
        )
        self.initialise_layer(self.conv1)
        self.norm1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=36,
            kernel_size=(5, 5),
            padding=(2, 2),
            stride=(2, 2),
            bias=False
        )
        self.initialise_layer(self.conv2)
        self.norm2 = nn.BatchNorm2d(36)

        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=48,
            kernel_size=(5, 5),
            padding=(2, 2),
            stride=(2,2),
            bias=False
        )
        self.initialise_layer(self.conv3)
        self.norm3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv4)
        self.norm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(
            in_channels=self.conv4.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv5)
        self.norm5 = nn.BatchNorm2d(64)

        size = int((image_height / 8) * (image_width / 8) * self.conv5.out_channels)

        self.fc1 = nn.Linear(size, 1164)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(1164, 100)
        self.initialise_layer(self.fc2)

        self.fc3 = nn.Linear(100, 50)
        self.initialise_layer(self.fc3)

        self.fc4 = nn.Linear(50, 10)
        self.initialise_layer(self.fc4)

        self.fc5 = nn.Linear(10, 1)
        self.initialise_layer(self.fc5)

    def forward(self, images) -> torch.Tensor:
        x = F.elu(self.norm1(self.conv1(images)))

        x = F.elu(self.norm2(self.conv2(x)))

        x = F.elu(self.norm3(self.conv3(x)))

        x = self.dropout1(x)

        x = F.elu(self.norm4(self.conv4(x)))

        x = F.elu(self.norm5(self.conv5(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.dropout2(x)

        x = F.elu(self.fc1(x))

        x = F.elu(self.fc2(x))

        x = F.elu(self.fc3(x))

        x = F.elu(self.fc4(x))

        x = self.fc5(x)

        print(x, flush=True)

        x = F.tanh(x / 100) * 45.0

        print(x, flush=True)

        return x

    @staticmethod
    def initialise_layer(layer):
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


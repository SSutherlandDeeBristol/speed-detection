# Local Modules
from bdd_dataset import BDDDataset
from cnn import CNN

# Python Modules
import time

# PyTorch Modules
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Input/Output Modules
import argparse
from pathlib import Path
import sys
import pickle

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def get_summary_writer_log_dir(dataset_type) -> str:
    tb_log_dir_prefix = (
        f'{dataset_type}_'
        f'run_'
    )
    i = 0
    while i < 1000:
        tb_log_dir = Path("logs") / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__=='__main__':
    dataset_type = 'original'

    log_dir = get_summary_writer_log_dir(dataset_type)

    summary_writer = SummaryWriter(
                str(log_dir),
                flush_secs=5
        )

    criterion = torch.nn.MSELoss()

    model = CNN(1280, 720, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

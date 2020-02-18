# Local Modules
from bdd_dataset import BDDDataset
from cnn import CNN
from trainer import Trainer

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

def get_summary_writer_log_dir() -> str:
    tb_log_dir_prefix = (
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
    log_dir = get_summary_writer_log_dir()

    summary_writer = SummaryWriter(
                str(log_dir),
                flush_secs=5
        )

    criterion = torch.nn.MSELoss()

    image_width = 640
    image_height = 360

    model = CNN(image_width, image_height, 3)

    resize_transform = transforms.Resize((image_height, image_width))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4)

    train_loader = torch.utils.data.DataLoader(
        BDDDataset('../../train/', 'dataset_train.pkl', resize_transform),
        batch_size=16, shuffle=True,
        num_workers=8, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        BDDDataset('../../val/', 'dataset_val.pkl', resize_transform),
        batch_size=16, shuffle=False,
        num_workers=8, pin_memory=True
    )

    trainer = Trainer(model,
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer,
                      summary_writer,
                      DEVICE)

    trainer.train(10,
                  2,
                  1,
                  1)

    trainer.save_model()

    print(log_dir)

    summary_writer.close()

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

    model = CNN(1280, 720, 3)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-3)

    train_loader = torch.utils.data.DataLoader(
        BDDDataset('~/../../scratch/ss15060/train/', 'dataset_train.pkl', None),
        batch_size=2, shuffle=True,
        num_workers=8, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        BDDDataset('~/../../scratch/ss15060/val/', 'dataset_val.pkl', None),
        batch_size=2, shuffle=False,
        num_workers=8, pin_memory=True
    )

    trainer = Trainer(model,
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer,
                      summary_writer,
                      DEVICE)

    trainer.train(20,
                  1,
                  1,
                  1)

    trainer.save_model()

    print(log_dir)

    summary_writer.close()

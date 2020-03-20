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
from torch.autograd import Variable

# Input/Output Modules
import argparse
from pathlib import Path
import sys
import pickle

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

parser = argparse.ArgumentParser(
    description="Train the CNN.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr',
                    default=1e-3,
                    type=float,
                    help="Learning Rate.")

parser.add_argument('--bs',
                    default=64,
                    type=int,
                    help="Batch size.")

def custom_loss(output, target):
    x = target - output
    loss = torch.mean(torch.Tensor(Variable([y**2 if y > 0 else (2*y)**2 for y in x], requires_grad=True)))
    return loss

def get_summary_writer_log_dir(batch_size, learning_rate) -> str:
    tb_log_dir_prefix = (
        f'bs_{batch_size}_'
        f'lr_{learning_rate}_'
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

    args = parser.parse_args()

    batch_size = args.bs
    learning_rate = args.lr

    log_dir = get_summary_writer_log_dir(batch_size, learning_rate)

    summary_writer = SummaryWriter(
                str(log_dir),
                flush_secs=5
        )

    #criterion = torch.nn.MSELoss()
    criterion = custom_loss

    image_width = 640
    image_height = 360

    model = CNN(image_width, image_height, 3)

    resize_transform = transforms.Resize((image_height, image_width))
    affine_transform = transforms.RandomAffine(degrees=15, translate=(0,0.2))
    perspective_transform = transforms.RandomPerspective(p=0.5, distortion_scale=0.5)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        BDDDataset('../../train/', 'dataset_train.pkl', transforms.Compose([resize_transform,
                                                                            affine_transform])),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        BDDDataset('../../val/', 'dataset_val.pkl', resize_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    trainer = Trainer(model,
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer,
                      summary_writer,
                      DEVICE)

    trainer.train(15,
                  1,
                  1,
                  1)

    trainer.save_model()

    print(log_dir)

    summary_writer.close()

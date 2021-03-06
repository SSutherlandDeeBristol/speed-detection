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

parser.add_argument('--epochs',
                    default=15,
                    type=int,
                    help="Number of epochs.")

def truncated_sum(output, target):
    x = output.sub(target)
    x = torch.clamp(x, -10, 10)

    pos_mask = x.ge(0)
    neg_mask = x.lt(0)

    pos_error = torch.masked_select(x, pos_mask)
    neg_error = torch.masked_select(x, neg_mask)

    pos_error = pos_error.pow(2)
    neg_error = torch.mul(neg_error.pow(2), 2)

    errors = torch.cat((pos_error, neg_error))

    loss = torch.sum(errors)

    return loss

def truncated_mse(output, target):
    x = output.sub(target)
    x = torch.clamp(x, -12.5, 12.5)
    x = x.pow(2)

    loss = torch.mean(x)

    return loss

def truncated_loss(output, target):
    x = output.sub(target)
    x = torch.clamp(x, -10, 10)

    pos_mask = x.ge(0)
    neg_mask = x.lt(0)

    pos_error = torch.masked_select(x, pos_mask)
    neg_error = torch.masked_select(x, neg_mask)

    pos_error = pos_error.pow(2)
    neg_error = torch.mul(neg_error.pow(2), 2)

    errors = torch.cat((pos_error, neg_error))

    loss = torch.mean(errors)

    return loss

def asymmetric_loss(output, target):
    x = output.sub(target)
    pos_mask = x.ge(0)
    neg_mask = x.lt(0)

    pos_error = torch.masked_select(x, pos_mask)
    neg_error = torch.masked_select(x, neg_mask)

    pos_error = pos_error.pow(2)
    neg_error = torch.mul(neg_error.pow(2), 2)

    errors = torch.cat((pos_error, neg_error))

    loss = torch.mean(errors)

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
    # Collect arguments
    args = parser.parse_args()

    batch_size = args.bs
    learning_rate = args.lr
    epochs = args.epochs

    log_dir = get_summary_writer_log_dir(batch_size, learning_rate)

    summary_writer = SummaryWriter(
                str(log_dir),
                flush_secs=5
        )

    # Huber Loss
    criterion = torch.nn.SmoothL1Loss()

    image_width = 640
    image_height = 360

    # Initialise CNN
    model = CNN(image_width, image_height, 3)

    # Initialise transforms
    resize_transform = transforms.Resize((image_height, image_width))
    affine_transform = transforms.RandomAffine(degrees=15, translate=(0,0.2))
    to_tensor_transform = transforms.ToTensor()

    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        BDDDataset('../../train/', 'dataset_train.pkl', transforms.Compose([resize_transform, affine_transform, to_tensor_transform])),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        BDDDataset('../../val/', 'dataset_test.pkl', transforms.Compose([resize_transform, to_tensor_transform])),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    # Initialise Trainer
    trainer = Trainer(model,
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer,
                      summary_writer,
                      DEVICE)

    # Train the model
    trainer.train(epochs,
                  1,
                  1,
                  1)

    # Save the model
    trainer.save_model()

    print(log_dir)

    summary_writer.close()

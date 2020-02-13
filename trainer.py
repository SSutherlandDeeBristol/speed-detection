import time
import pickle as pkl

from multiprocessing import cpu_count
from typing import Union, NamedTuple
import math
import os

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

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device
      ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.logit_dir = os.path.join(self.summary_writer.log_dir, "logits")
        os.mkdir(self.logit_dir)

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
    ):
        self.model.train()

        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()

            for batch, labels, in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                batch = batch.float()
                labels = labels.float()

                logits = self.model.forward(batch)

                labels = np.expand_dims(labels,axis=1)

                loss = self.criterion(logits, torch.Tensor(labels))

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch,
                                     loss,
                                     data_load_time,
                                     step_time)

                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch,
                                       loss,
                                       data_load_time,
                                       step_time)
                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate(epoch)
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}",
                flush=True
        )

    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )


    def validate(self, epoch_num):
        self.model.eval()

        total_loss = 0

        base_address = 0
        logit_log = {}

        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                batch = batch.float()
                labels = labels.float()

                logits = self.model(batch)

                labels = np.expand_dims(labels,axis=1)

                loss = self.criterion(logits, torch.Tensor(labels))

                total_loss += loss.item()

                for j in range(batch.shape[0]):
                    logit_log[base_address + j] = (logits[j].item(),
                        labels[j].item()
                    )
                base_address += 1

        self.save_logits_to_logs(epoch_num, logit_log)

        average_loss = total_loss / len(self.val_loader)

        self.print_validation_metrics(total_loss,
                                      average_loss,
                                      epoch_num)

        self.log_validation_metrics(average_loss)

    def print_validation_metrics(self, loss, average_loss, epoch_num):
        print(f"loss: {loss:2.2f}", flush=True)

        print(f"average loss: {average_loss:.5f}", flush=True)

        print(f"epoch: {epoch_num}")

    def log_validation_metrics(self, average_loss):
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )

    def save_logits_to_logs(self, epoch, log):
        epoch_dir = os.path.join(self.logit_dir, str(epoch) +".pkl")
        with open(epoch_dir, 'wb') as handle:
                pkl.dump(log, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def save_model(self):
        file_path = os.path.join(self.summary_writer.log_dir, "model.pt")
        torch.save(self.model.state_dict(), file_path)

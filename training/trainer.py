import os
import sys

sys.path.append("..")  # cd xx/playground

import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from models import modelZoo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.IOutils import save_checkpoint
from utils.logger import setup_logger
from utils.parallelWraper import CustomDataDistParallel as DDP
from dataset.trainDataset import nyuv2, sun


class Trainer:
    def __init__(self, args, model_config) -> None:
        self.cur_loss = 0
        self.best_loss = 1e10
        self.device = "cuda"
        self.dist = args.dist
        self.rank = args.local_rank
        self.epochs = args.epochs
        self.clip_max_norm = args.clip_max_norm

        self.channel = args.channel
        self.quality = args.quality

        if args.experiment is not None and args.experiment != "":
            self.exp_name = args.experiment
        else:
            self.exp_name = self.get_exp_name(args.dataset, args.channel, args.model, args.quality)

        self.exp_dir_path = os.path.join("../experiments", self.exp_name)
        self.ckpt_dir_path = os.path.join(self.exp_dir_path, "checkpoints")
        self.init_dir([self.exp_dir_path, self.ckpt_dir_path])
        self.logger_train, self.logger_val, self.tb_logger = self.init_logger(self.exp_dir_path, self.exp_name)

        self.model_name = args.model
        for name, model in modelZoo.items():
            if args.model.find(name) != -1:
                self.net = model(config=model_config, channel=args.channel)
                self.logger_train.info(f"model name: {name}")
                break
        if self.rank == 0:
            self.logger_train.info(args)
            # self.logger_train.info(self.net)
        self.train_dataloader, self.val_dataloader = self.init_dataset(
            args.dataset, args.val_dataset, args.batch_size, args.test_batch_size, args.num_workers
        )
        self.learning_rate = args.learning_rate
        self.aux_learning_rate = args.aux_learning_rate
        self.optimizer, self.aux_optimizer = self.configure_optimizers()
        self.lr_scheduler = self.get_lr_scheduler(args.lr_scheduler)

    def get_exp_name(self, dataset=None, channel=3, model_name=None, quality=1):
        if channel == 1:
            modal = "depth_"
        elif channel == 3:
            modal = "rgb_"
        elif channel == 4:
            modal = ""
        dataset_name = self.get_dataset_name(dataset)
        exp_name = f"{dataset_name}_{modal}{model_name}_{quality}"
        return exp_name

    def get_dataset_name(self, dataset):
        if dataset.find("nyu") != -1:
            return "nyuv2"
        return "sunrgbd"

    def configure_optimizers(self):
        """Separate parameters for the main optimizer and the auxiliary optimizer.
        Return two optimizers"""

        parameters = {n for n, p in self.net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
        aux_parameters = {n for n, p in self.net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.net.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=self.learning_rate)
        aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=self.aux_learning_rate)
        return optimizer, aux_optimizer

    def get_lr_scheduler(self, ls_name):
        if ls_name == "ReduceLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        elif ls_name == "MultiStepLR":
            milestones = [int(self.epochs * 0.75), int(self.epochs * 0.9)]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        return lr_scheduler

    def init_dataset(self, train_dataset, val_dataset, batch_size, test_batch_size, num_workers):
        Dataset = nyuv2 if train_dataset.find("nyu") != -1 else sun
        train_dataset = Dataset(train_dataset, is_train=True, channel=self.channel)
        val_dataset = Dataset(val_dataset, is_train=False, channel=self.channel)

        is_shuffle = True
        if self.dist:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            is_shuffle = False
        else:
            train_sampler, val_sampler = None, None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=is_shuffle,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True, sampler=val_sampler
        )
        return train_dataloader, val_dataloader

    def init_dir(self, dirs):
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def setup_seed(self, seed=20):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_logger(self, exp_dir_path, exp_name):
        setup_logger("train", exp_dir_path, "train_" + exp_name)
        setup_logger("val", exp_dir_path, "val_" + exp_name)
        logger_train = logging.getLogger("train")
        logger_val = logging.getLogger("val")
        tb_logger = SummaryWriter(log_dir="../tb_logger/" + exp_name)
        return logger_train, logger_val, tb_logger

    def restore(self, ckpt_path=None):
        checkpoint = torch.load(ckpt_path)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.net.update(force=True)
        self.net = self.net.to(self.device)
        return checkpoint["epoch"]

    def fit(self, seed=None, auto_restore=False, ckpt_path=None, start_epoch=0):
        if seed is not None:
            self.setup_seed(seed)

        if auto_restore and os.path.exists(os.path.join(self.ckpt_dir_path, "checkpoint_best_loss.pth.tar")):
            ckpt_path = os.path.join(self.ckpt_dir_path, "checkpoint_best_loss.pth.tar")
        if ckpt_path:
            start_epoch = self.restore(ckpt_path)

        self.net = self.net.to(self.device)
        if self.dist:
            self.net = DDP(self.net, device_ids=[self.rank])

        current_step = 0  # tensorboard
        for epoch in range(start_epoch, self.epochs):
            if self.rank == 0:
                self.logger_train.info(f"{self.exp_dir_path} Epoch:{epoch}, lr: {self.optimizer.param_groups[0]['lr']}")
            current_step = self.train_one_epoch(epoch, current_step, clip_max_norm=self.clip_max_norm)
            self.cur_loss = self.validate_one_epoch(epoch)
            if isinstance(self.lr_scheduler, optim.lr_scheduler.MultiStepLR):
                self.lr_scheduler.step()
            elif isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.cur_loss)
            self.save_ckpt(epoch)

    def save_ckpt(self, epoch, every_epoch=200):
        is_best = self.cur_loss < self.best_loss
        self.best_loss = min(self.cur_loss, self.best_loss)
        ckpt = {
            "epoch": epoch + 1,
            "state_dict": self.net.module.state_dict() if self.dist else self.net.state_dict(),
            "loss": self.cur_loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        latest_path = os.path.join(self.ckpt_dir_path, "checkpoint_latest.pth.tar")
        save_checkpoint(ckpt, is_best, latest_path)

        if (epoch + 1) % every_epoch == 0:
            epoch_path = str(latest_path).replace("latest", f"epoch{epoch}")
            save_checkpoint(ckpt, is_best, epoch_path)
        if is_best and self.rank == 0:
            self.logger_val.info(f"epoch:{epoch + 1} best checkpoint saved.")

    def train_one_epoch(self, epoch, current_step, clip_max_norm=1):
        pass

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        pass


if __name__ == "__main__":
    pass

# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
import random
import shutil
import sys

sys.path.append("/data/chenminghui/ELIC")

import logging
import pickle
import random
import time
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from config.args import train_options
from config.config import model_config
from models.elic import ELIC
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToPILImage
from utils import util

from elic_master import ELIC_master
from elic_single import ELIC_single
from utils_edge import nyuv2

# from compressai.zoo.image import model_architectures as architectures

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
torch.backends.cudnn.benchmark = True  # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
torch.backends.cudnn.deterministic = True  # 由于计算中有随机性，每次网络前馈结果略有差异，通过设置该flag避免这种波动


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1))


def compute_metrics(a: Union[np.array, Image.Image], b: Union[np.array, Image.Image], max_val: float = 1) -> Tuple[float, float]:  # 会自动转换
    """Returns PSNR and MS-SSIM between images `a` and `b`."""
    # if isinstance(a, Image.Image):
    #     a = np.asarray(a)
    # if isinstance(b, Image.Image):
    #     b = np.asarray(b)

    # a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    # if a.size(3) == 3:
    #     a = a.permute(0, 3, 1, 2)
    # b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    # if b.size(3) == 3:
    #     b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())
        if self.metrics == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == "ms-ssim":
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = [p for n, p in net.named_parameters() if not n.endswith(".quantiles")]
    aux_parameters = [p for n, p in net.named_parameters() if n.endswith(".quantiles")]

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = set(parameters) & set(aux_parameters)
    union_params = set(parameters) | set(aux_parameters)

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam((p for p in parameters if p.requires_grad), lr=args.learning_rate)
    aux_optimizer = optim.Adam((p for p in aux_parameters if p.requires_grad), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer


def train_one_epoch(model, model_single, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        aux = d[0]
        aux = aux.to(device)
        d = d[1]
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with torch.no_grad():
            out = model_single(aux)
        out_net = model(d, out["x_hat"], out)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar("{}".format("[train]: loss"), out_criterion["loss"].item(), current_step)
            tb_logger.add_scalar("{}".format("[train]: bpp_loss"), out_criterion["bpp_loss"].item(), current_step)
            if out_criterion["mse_loss"] is not None:
                tb_logger.add_scalar("{}".format("[train]: mse_loss"), out_criterion["mse_loss"].item(), current_step)
            if out_criterion["ms_ssim_loss"] is not None:
                tb_logger.add_scalar("{}".format("[train]: ms_ssim_loss"), out_criterion["ms_ssim_loss"].item(), current_step)

        if i % 100 == 0:
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step


def test_epoch(epoch, test_dataloader, model, model_single, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            aux = d[0]
            aux = aux.to(device)
            d = d[1]
            d = d.to(device)
            out = model_single(aux)
            out_net = model(d, out["x_hat"], out)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            # for num in range(len(d)):
            # 转成图像应该很费时间吧
            # rec = torch2img(out_net['x_hat'][num])
            # img = torch2img(d[num])
            # p, m = compute_metrics(rec, img)
            rec = out_net["x_hat"].clamp_(0, 1)
            img = d
            p, m = compute_metrics(rec, img)

            psnr.update(p)
            ms_ssim.update(m)

            if i % 20 == 1:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                rec = torch2img(rec[0])
                img = torch2img(img[0])
                rec.save(os.path.join(save_dir, "%03d_rec.png" % i))
                img.save(os.path.join(save_dir, "%03d_gt.png" % i))

    tb_logger.add_scalar("{}".format("[val]: loss"), loss.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: bpp_loss"), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: psnr"), psnr.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: ms-ssim"), ms_ssim.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.2f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar("{}".format("[val]: mse_loss"), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.2f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar("{}".format("[val]: ms_ssim_loss"), ms_ssim_loss.avg, epoch + 1)

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(filename.split("/")[-1], "checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, dest_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-exp", "--experiment", type=str, default="", help="Experiment name")
    parser.add_argument("-m", "--model", default="ELIC", help="Model architecture (default: %(default)s)")
    # parser.add_argument(
    #     "-d", "--dataset", type=str, required=True, help="Training dataset"
    # )  # 直接指定
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")

    parser.add_argument("-e", "--epochs", default=400, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("-q", "--quality", type=int, default=1, help="Quality (default: %(default)s)")
    parser.add_argument("--local_rank", type=int, default=0, help="Quality (default: %(default)s)")
    # parser.add_argument(
    #     "-n",
    #     "--num-workers",
    #     type=int,
    #     default=30,
    #     help="Dataloaders threads (default: %(default)s)",
    # ) # 直接指定
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument("--metrics", type=str, default="mse", help="Optimized for (default: %(default)s)")  # 直接指定
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")  # 直接指定
    # parser.add_argument(
    #     "--test-batch-size",
    #     type=int,
    #     default=1,
    #     help="Test batch size (default: %(default)s)",
    # ) # 直接指定
    parser.add_argument("--aux-learning-rate", default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Size of the patches to be cropped (default: %(default)s)")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm (default: %(default)s")
    parser.add_argument("-c", "--checkpoint", default=None, type=str, help="pretrained model path")
    parser.add_argument("-c1", "--checkpoint1", default=None, type=str, help="pretrained model path")
    parser.add_argument("--git", action="store_true", help="Use git to save code")
    parser.add_argument("--restore", action="store_true", help="Restore ckt automatically")
    args = parser.parse_args(argv)
    return args


def main(argv):
    config = model_config()
    args = parse_args(argv)
    args.milestones = [int(args.epochs * 0.75), int(args.epochs * 0.9)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    train_dir = "/data/chenminghui/nyud/nyu5k"
    val_dir = "/data/chenminghui/nyud/nyu5k/val"
    args.channel = 1
    lmbdas = [0.00180, 0.00350, 0.00670, 0.01300, 0.02500, 0.04830, 0.09320, 0.1800]
    args.lmbda = lmbdas[args.quality]
    if args.experiment == "":
        args.experiment = f"nyuv2_depth_{args.model}_{args.quality}"
        print("exp name:", args.experiment)
    if args.git:
        git_add = "git add ."
        git_commit = f"git commit -m {args.experiment}"
        os.system(git_add)
        os.system(git_commit)
        print(git_add, git_commit)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not os.path.exists(os.path.join("../experiments", args.experiment)):
        os.makedirs(os.path.join("../experiments", args.experiment))

    util.setup_logger("train", os.path.join("../experiments", args.experiment), "train_" + args.experiment, level=logging.INFO, screen=True, tofile=True)
    util.setup_logger("val", os.path.join("../experiments", args.experiment), "val_" + args.experiment, level=logging.INFO, screen=True, tofile=True)

    logger_train = logging.getLogger("train")
    logger_val = logging.getLogger("val")
    tb_logger = SummaryWriter(log_dir="../tb_logger/" + args.experiment)

    if not os.path.exists(os.path.join("../experiments", args.experiment, "checkpoints")):
        os.makedirs(os.path.join("../experiments", args.experiment, "checkpoints"))

    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    # train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    # test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataset = nyuv2(train_dir, True)
    test_dataset = nyuv2(val_dir, False)
    device = "cuda"

    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=2, shuffle=True, pin_memory=(device == "cuda"), drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=(device == "cuda"))

    # 自动装载
    ckt_path = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_best_loss.pth.tar")
    if os.path.exists(ckt_path) and args.restore:
        args.checkpoint = ckt_path

    net = ELIC_master(config=config, ch=1)
    net_single = ELIC_single(config=config, ch=3)
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint["state_dict"])
        net.update(force=True)
    net_single_ckt = torch.load(args.checkpoint1)
    net_single.load_state_dict(net_single_ckt["state_dict"])
    net_single.update(force=True)
    net_single.eval()
    net = net.to(device)
    net_single = net_single.to(device)

    logger_train.info(args)
    logger_train.info(net)

    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        net_single = CustomDataParallel(net_single)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)

    if args.checkpoint != None:
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        lr_scheduler._step_count = checkpoint["lr_scheduler"]["_step_count"]
        lr_scheduler.last_epoch = checkpoint["lr_scheduler"]["last_epoch"]
        # print(lr_scheduler.state_dict())
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        checkpoint = None
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(net, net_single, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, logger_train, tb_logger, current_step)

        save_dir = os.path.join("../experiments", args.experiment, "val_images", "%03d" % (epoch + 1))
        loss = test_epoch(epoch, test_dataloader, net, net_single, criterion, save_dir, logger_val, tb_logger)
        # lr_scheduler.step(loss)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                # os.path.join('../experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
                os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_latest.pth.tar"),  # 存储空间不够
            )
            if (epoch + 1) % 200 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    # os.path.join('../experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
                    os.path.join("../experiments", args.experiment, "checkpoints", f"checkpoint_epoch{epoch}.pth.tar"),  # 存储空间不够
                )
            if is_best:
                logger_val.info(f"epoch:{epoch + 1} best checkpoint saved.")


if __name__ == "__main__":
    main(sys.argv[1:])

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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from compressai.datasets import ImageFolder
from compressai.zoo import models
from config.args import train_options
from config.config import model_config
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToPILImage
from utils import util

from elic_united4 import ELIC_united
# from models.elic import ELIC
from elic_united_cpf import ELIC_united_cpf
from elic_united_EEM1 import ELIC_united_EEM
from loss import ssim
from utils_edge import nyuv2

RANK = 0
# from compressai.zoo.image import model_architectures as architectures

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
torch.backends.cudnn.benchmark = True  # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
torch.backends.cudnn.deterministic = True  # 由于计算中有随机性，每次网络前馈结果略有差异，通过设置该flag避免这种波动 # 两者一般设置为相反，具有一定的冲突


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1))


def safe_torch_save(obj, filename):
    try:
        # 执行 torch.save 操作
        torch.save(obj, filename)

        # 显式关闭文件以确保数据完全写入磁盘
        if os.path.exists(filename):
            with open(filename, "rb"):
                pass
    except Exception as e:
        print(f"Error while saving: {e}")


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


def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, quality):
        super().__init__()
        self.mse = nn.MSELoss()
        lmbdas = [0.00180, 0.00350, 0.00670, 0.01300, 0.02500, 0.04830, 0.09320, 0.1800]
        rgb_q, depth_q = quality.split("_")
        rgb_q = float(rgb_q)
        depth_q = float(depth_q)
        if int(rgb_q) == rgb_q:
            self.rgb_lmbda = lmbdas[int(rgb_q)]
        else:
            self.rgb_lmbda = (lmbdas[math.ceil(rgb_q)] + lmbdas[math.floor(rgb_q)]) / 2

        if int(depth_q) == depth_q:
            self.depth_lmbda = lmbdas[int(depth_q)]
        else:
            self.depth_lmbda = (lmbdas[math.ceil(depth_q)] + lmbdas[math.floor(depth_q)]) / 2

        self.l1_criterion = nn.L1Loss()

    def get_final_loss(self, out):
        # out['loss'] = (out['ssim_loss'] + out['edge_loss'] + 0.1 * out['l1_loss']) * self.lmbda + out["bpp_loss"]
        # out["d_mse_loss"] = self.mse(d, depth)

        # out["loss"] = self.lmbda * 255 ** 2 * (out["r_mse_loss"]+out["d_mse_loss"]) + out["r_bpp_loss"] + out["d_bpp_loss"]
        out["loss"] = self.rgb_lmbda * 255**2 * out["r_mse_loss"] + out["r_bpp_loss"] + self.depth_lmbda * 255**2 * 0.01 * out["d_loss"] + out["d_bpp_loss"]
        return out

    def forward(self, output, rgb, depth):
        N, _, H, W = rgb.size()
        out = {}
        num_pixels = N * H * W

        # 模拟bpp
        out["r_bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["r_likelihoods"].values())
        out["d_bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["d_likelihoods"].values())

        # rgb loss
        r = output["x_hat"]["r"]
        out["r_mse_loss"] = self.mse(r, rgb)

        # depth loss
        d = output["x_hat"]["d"]
        out["d_mse_loss"] = self.mse(d, depth)  # mse可能会导致图像平滑，不适用
        out["l1_loss"] = self.l1_criterion(d, depth)

        output_dx, output_dy = gradient(d)
        target_dx, target_dy = gradient(depth)
        grad_diff_x = torch.abs(output_dx - target_dx)
        grad_diff_y = torch.abs(output_dy - target_dy)
        out["edge_loss"] = torch.mean(grad_diff_x + grad_diff_y)

        out["ssim_loss"] = torch.clamp((1 - ssim(d, depth, val_range=1)) * 0.5, 0, 1)
        out["d_loss"] = out["ssim_loss"] + out["edge_loss"] + 0.1 * out["l1_loss"]

        return self.get_final_loss(out)


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


class CustomDataDistParallel(nn.parallel.DistributedDataParallel):
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


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        rgb, depth = d[0], d[1]
        d = d[0]
        rgb, depth = rgb.to(device), depth.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(rgb, depth)
        out_criterion = criterion(out_net, rgb, depth)

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
            tb_logger.add_scalar("{}".format("[train]: rbpp_loss"), out_criterion["r_bpp_loss"].item(), current_step)
            tb_logger.add_scalar("{}".format("[train]: dbpp_loss"), out_criterion["d_bpp_loss"].item(), current_step)
            tb_logger.add_scalar("{}".format("[train]: rmse_loss"), out_criterion["r_mse_loss"].item(), current_step)
            tb_logger.add_scalar("{}".format("[train]: d_loss"), out_criterion["d_loss"].item(), current_step)

        if i % 100 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.4f} | '
                f'rMSE loss: {out_criterion["r_mse_loss"].item():.4f} | '
                f'rBpp loss: {out_criterion["r_bpp_loss"].item():.4f} | '
                f'dBpp loss: {out_criterion["d_bpp_loss"].item():.4f} | '
                f'd loss: {out_criterion["d_loss"].item():.2f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )
    return current_step


def test_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    r_bpp_loss = AverageMeter()
    d_bpp_loss = AverageMeter()
    r_mse_loss = AverageMeter()
    d_loss = AverageMeter()
    aux_loss = AverageMeter()
    r_psnr = AverageMeter()
    d_psnr = AverageMeter()
    r_ms_ssim = AverageMeter()
    d_ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            rgb, depth = d[0], d[1]
            d = d[1]
            rgb, depth = rgb.to(device), depth.to(device)
            out_net = model(rgb, depth)
            out_criterion = criterion(out_net, rgb, depth)

            r_bpp_loss.update(out_criterion["r_bpp_loss"])
            d_bpp_loss.update(out_criterion["d_bpp_loss"])
            loss.update(out_criterion["loss"])
            r_mse_loss.update(out_criterion["r_mse_loss"])
            d_loss.update(out_criterion["d_loss"])

            rec_r = out_net["x_hat"]["r"].clamp_(0, 1)
            p, m = compute_metrics(rec_r, rgb)
            r_psnr.update(p)
            r_ms_ssim.update(m)

            rec_d = out_net["x_hat"]["d"].clamp_(0, 1)
            p, m = compute_metrics(rec_d, depth)
            d_psnr.update(p)
            d_ms_ssim.update(m)

            if i % 20 == 1 and RANK == 0:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                rec_r = torch2img(rec_r[0])
                # img = torch2img(img[0])
                rec_r.save(os.path.join(save_dir, "%03d_rgb_rec.png" % i))
                # img.save(os.path.join(save_dir, '%03d_gt.png' % i))

                rec_d = torch2img(rec_d[0])
                rec_d.save(os.path.join(save_dir, "%03d_depth_rec.png" % i))

    tb_logger.add_scalar("{}".format("[val]: loss"), loss.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: rbpp_loss"), r_bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: dbpp_loss"), d_bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: rpsnr"), r_psnr.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: dpsnr"), d_psnr.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: rms-ssim"), r_ms_ssim.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: dms-ssim"), d_ms_ssim.avg, epoch + 1)

    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"rMSE loss: {r_mse_loss.avg:.4f} | "
        f"d loss: {d_loss.avg:.4f} | "
        f"rBpp loss: {r_bpp_loss.avg:.4f} | "
        f"dBpp loss: {d_bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} | "
        f"rPSNR: {r_psnr.avg:.4f} | "
        f"dPSNR: {d_psnr.avg:.4f} | "
        f"rMS-SSIM: {r_ms_ssim.avg:.4f} |"
        f"dMS-SSIM: {d_ms_ssim.avg:.4f}\n"
    )
    tb_logger.add_scalar("{}".format("[val]: r_mse_loss"), r_mse_loss.avg, epoch + 1)
    tb_logger.add_scalar("{}".format("[val]: d_loss"), d_loss.avg, epoch + 1)

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    safe_torch_save(state, filename)
    if is_best:
        dest_filename = filename.replace(filename.split("/")[-1], "checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, dest_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-exp", "--experiment", type=str, default="", help="Experiment name")
    parser.add_argument("-m", "--model", default="ELIC_cpf", help="Model architecture (default: %(default)s)")
    # parser.add_argument(
    #     "-d", "--dataset", type=str, required=True, help="Training dataset"
    # )  # 直接指定
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")

    parser.add_argument("-e", "--epochs", default=400, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("-q", "--quality", type=str, default="3_3", help="Quality")
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

    parser.add_argument("--batch_size", type=int, default=6, help="Batch size (default: %(default)s)")  # 直接指定
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
    parser.add_argument("--git", action="store_true", help="Use git to save code")
    parser.add_argument("--restore", action="store_true", help="Restore ckt automatically")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dist", action="store_true")
    args = parser.parse_args(argv)
    return args


def main(argv):
    config = model_config()
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.dist:  # 如果使用launch启动，但是没有使用dist参数，会有问题；启动了之后，出现grad的rank不对，
        dist.init_process_group(backend="nccl")  # wordsize是否会自己计算？
        torch.cuda.set_device(args.local_rank)  # 必须添加
    else:
        args.local_rank = 0  # 兼容
    global RANK
    RANK = args.local_rank
    args.milestones = [int(args.epochs * 0.75), int(args.epochs * 0.9)]

    train_dir = "/data/chenminghui/nyud/nyu5k"
    val_dir = "/data/chenminghui/nyud/nyu5k/val"

    if args.experiment == "":
        args.experiment = f"nyuv2_{args.model}_{args.quality}"
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
        os.makedirs(os.path.join("../experiments", args.experiment), exist_ok=True)

    util.setup_logger("train", os.path.join("../experiments", args.experiment), "train_" + args.experiment, level=logging.INFO, screen=True, tofile=True)
    util.setup_logger("val", os.path.join("../experiments", args.experiment), "val_" + args.experiment, level=logging.INFO, screen=True, tofile=True)

    logger_train = logging.getLogger("train")
    logger_val = logging.getLogger("val")
    tb_logger = SummaryWriter(log_dir="../tb_logger/" + args.experiment)

    if not os.path.exists(os.path.join("../experiments", args.experiment, "checkpoints")):
        os.makedirs(os.path.join("../experiments", args.experiment, "checkpoints"), exist_ok=True)

    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    # train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    # test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataset = nyuv2(train_dir, True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.dist else None
    test_dataset = nyuv2(val_dir, False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.dist else None
    device = "cuda"

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6, pin_memory=(device == "cuda"), drop_last=True, sampler=train_sampler)

    test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=4, shuffle=False, pin_memory=(device == "cuda"), sampler=test_sampler)

    # 自动装载
    ckt_path = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_best_loss.pth.tar")
    ckt_path1 = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_latest.pth.tar")
    if os.path.exists(ckt_path) and args.restore and not args.checkpoint:
        args.checkpoint = ckt_path
    elif os.path.exists(ckt_path1) and args.restore and not args.checkpoint:
        args.checkpoint = ckt_path1
    if args.model.find("ELIC_cpf") != -1:
        net = ELIC_united_cpf(config=config)
    elif args.model.find("ELIC_EEM") != -1:
        net = ELIC_united_EEM(config=config)
    elif args.model.find("ELIC_united") != -1:
        net = ELIC_united(config=config)

    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        net.update(force=True)
    net = net.to(device)
    if args.dist:
        net = CustomDataDistParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)

    logger_train.info(args)
    logger_train.info(net)

    # if torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    criterion = RateDistortionLoss(quality=args.quality)

    if args.checkpoint != None:
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        lr_scheduler._step_count = checkpoint["lr_scheduler"]["_step_count"]
        lr_scheduler.last_epoch = checkpoint["lr_scheduler"]["last_epoch"]
        # print(lr_scheduler.state_dict())
        start_epoch = checkpoint["epoch"]
        if args.checkpoint.find("best") != -1:
            best_loss = checkpoint["loss"]
        else:
            best_loss = 1e10
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        checkpoint = None
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, logger_train, tb_logger, current_step)

        save_dir = os.path.join("../experiments", args.experiment, "val_images", "%03d" % (epoch + 1))
        loss = test_epoch(epoch, test_dataloader, net, criterion, save_dir, logger_val, tb_logger)
        # lr_scheduler.step(loss)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if args.save:
            netdict = net.state_dict() if not args.dist else net.module.state_dict()
            save_checkpoint(
                {"epoch": epoch + 1, "state_dict": netdict, "loss": loss, "optimizer": optimizer.state_dict(), "aux_optimizer": aux_optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()},
                is_best,
                # os.path.join('../experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
                os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_latest.pth.tar"),  # 存储空间不够
            )
            if (epoch + 1) % 200 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": netdict,
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

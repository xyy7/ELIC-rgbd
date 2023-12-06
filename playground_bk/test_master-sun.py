import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("/home/xyy/ELIC")
import faulthandler

from config.args import test_options
from config.config import model_config
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import ImageFolder, ImageFolderUnited
from utils.logger import setup_logger
from utils.testing import test_master, test_model

from elic_master import ELIC_master
from elic_single import ELIC_single

faulthandler.enable()


def main():
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options()
    config = model_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.dataset = "/data/xyy/sunrgbd/test"
    args.channel = 3 if args.split == "rgb" else 1
    args.model = "ELIC_master"
    if args.experiment == "":
        args.experiment = f"sunrgbd_{args.split}_{args.model}_{args.quality}"
        print("exp name:", args.experiment)

    # 自动装载
    ckt_path1 = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_best_loss.pth.tar")
    ckt_path = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_latest.pth.tar")
    if os.path.exists(ckt_path) and not args.checkpoint:
        args.checkpoint = ckt_path
    elif os.path.exists(ckt_path1) and not args.checkpoint:
        args.checkpoint = ckt_path1
    print("ckpt", args.checkpoint)

    # logger增加epoch名称
    checkpoint = torch.load(args.checkpoint)
    if args.checkpoint != None:
        epoch = checkpoint["epoch"]

    device = "cuda"
    if not os.path.exists(os.path.join("../experiments", args.experiment)):
        os.makedirs(os.path.join("../experiments", args.experiment))
    padding_mode = "replicate0"
    setup_logger("test", os.path.join("../experiments", args.experiment), f"test_epoch{epoch}" + args.experiment + " " + padding_mode, level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger("test")

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageFolderUnited(args.dataset, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)

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
    net.eval()
    net = net.to(device)
    net_single = net_single.to(device)

    logger_test.info(f"Start testing!")
    save_dir = os.path.join("../experiments", args.experiment, "codestream", "%02d" % (epoch + 1))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_master(net=net, net_single=net_single, test_dataloader=test_dataloader, logger_test=logger_test, save_dir=save_dir, epoch=epoch, mode=padding_mode)


def set_free_cpu(rate=0.1, need_cpu=15):
    import os

    import psutil

    cpuinfo = psutil.cpu_percent(interval=0.5, percpu=True)
    freecpu = []
    for i, cinfo in enumerate(cpuinfo):
        if cinfo > rate:
            continue
        freecpu.append(i)
    os.sched_setaffinity(os.getpid(), freecpu[-need_cpu:])


if __name__ == "__main__":
    main()

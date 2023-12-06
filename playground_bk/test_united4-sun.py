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
from models import ELIC
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import ImageFolderUnited
from utils.logger import setup_logger
from utils.testing import test_model_united

from elic_united4 import ELIC_united
from elic_united_cpf import ELIC_united_cpf
from elic_united_EEM1 import ELIC_united_EEM

faulthandler.enable()


def main():
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options()
    config = model_config()
    args.dataset = "/data/xyy/sunrgbd/test"  # 修改[因为已经修改成符合nyuv2的读取方式了,所以不需要改]
    if args.experiment == "":
        args.experiment = f"sunrgbd_{args.model}_{args.quality}"
        print("exp name:", args.experiment)

    # 自动装载
    # 自动装载
    ckt_path1 = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_best_loss.pth.tar")
    ckt_path = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_latest.pth.tar")
    if os.path.exists(ckt_path) and not args.checkpoint:
        args.checkpoint = ckt_path
    elif os.path.exists(ckt_path1) and not args.checkpoint:
        args.checkpoint = ckt_path1
    device = "cuda"
    print(args.checkpoint)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # logger增加epoch名称
    try:
        print("using best checkpoint.")
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except:
        print("using laest checkpoint.")
        checkpoint = torch.load(ckt_path1, map_location=device)

    if args.checkpoint != None:
        epoch = checkpoint["epoch"]
    print("epoch:", epoch)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not os.path.exists(os.path.join("../experiments", args.experiment)):
        os.makedirs(os.path.join("../experiments", args.experiment))
    # padding_mode = "constant0"
    # padding_mode = "constant1"
    # padding_mode = "reflect0"
    # padding_mode = "reflect1"
    padding_mode = "replicate0"
    # padding_mode = "replicate1"
    # for padding_mode in ["constant0", "constant1", "reflect0", "reflect1", "replicate0", "replicate1"]:
    setup_logger("test", os.path.join("../experiments", args.experiment), f"test_epoch{epoch}" + args.experiment + "_" + padding_mode, level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger("test")

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageFolderUnited(args.dataset, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=0, shuffle=False)

    if args.model.find("ELIC_cpf") != -1:
        net = ELIC_united_cpf(config=config)
    elif args.model.find("ELIC_EEM") != -1:
        net = ELIC_united_EEM(config=config)
    elif args.model.find("ELIC_united") != -1:
        net = ELIC_united(config=config)

    net = net.to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.update(force=True)
    epoch = checkpoint["epoch"]
    logger_test.info(f"Start testing!")
    save_dir = os.path.join("../experiments", args.experiment, "codestream", "%02d" % (epoch + 1))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model_united(net=net, test_dataloader=test_dataloader, logger_test=logger_test, save_dir=save_dir, epoch=epoch, mode=padding_mode)


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
    # set_free_cpu()
    main()

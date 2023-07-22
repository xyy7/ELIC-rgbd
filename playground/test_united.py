import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("/data/chenminghui/ELIC")
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

from elic_united import ELIC_united
from elic_united_cpf import ELIC_united_cpf
from elic_united_EEM import ELIC_united_EEM

faulthandler.enable()


def main():
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options()
    config = model_config()
    if args.experiment == "":
        args.experiment = f"nyuv2_{args.model}_{args.quality}"
        print("exp name:", args.experiment)

    # 自动装载
    ckt_path = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_best_loss123.pth.tar")
    ckt_path1 = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_latest.pth.tar")
    if os.path.exists(ckt_path):
        args.checkpoint = ckt_path
    elif os.path.exists(ckt_path1):
        args.checkpoint = ckt_path1
    device = "cuda"
    print(args.checkpoint)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # logger增加epoch名称
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.checkpoint != None:
        epoch = checkpoint["epoch"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not os.path.exists(os.path.join("../experiments", args.experiment)):
        os.makedirs(os.path.join("../experiments", args.experiment))
    setup_logger("test", os.path.join("../experiments", args.experiment), f"test_epoch{epoch}" + args.experiment, level=logging.INFO, screen=True, tofile=True)
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
    test_model_united(net=net, test_dataloader=test_dataloader, logger_test=logger_test, save_dir=save_dir, epoch=epoch)


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
    set_free_cpu()
    main()

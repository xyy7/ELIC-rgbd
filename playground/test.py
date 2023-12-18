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
from utils.dataset import ImageFolder
from utils.logger import setup_logger
from utils.testing import test_model

faulthandler.enable()


def main(argv):
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options(argv)
    config = model_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.channel = 3 if args.split == "rgb" else 1
    if args.experiment == "":
        args.experiment = f"nyuv2_{args.split}_{args.model}_{args.quality}"
        print("exp name:", args.experiment)

    ckt_path = os.path.join("../experiments", args.experiment, "checkpoints", "checkpoint_best_loss.pth.tar")
    if os.path.exists(ckt_path) and not args.checkpoint:
        args.checkpoint = ckt_path

    # logger增加epoch名称
    checkpoint = torch.load(args.checkpoint)
    if args.checkpoint != None:
        epoch = checkpoint["epoch"]

    device = "cuda"
    if not os.path.exists(os.path.join("../experiments", args.experiment)):
        os.makedirs(os.path.join("../experiments", args.experiment))
    padding_mode = "replicate0"
    setup_logger(
        "test",
        os.path.join("../experiments", args.experiment),
        f"test_epoch{epoch}" + args.experiment + " " + padding_mode,
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger_test = logging.getLogger("test")

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageFolder(args.dataset, split=args.split, transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False
    )

    net = ELIC(config=config, ch=args.channel)
    net = net.to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.update(force=True)
    logger_test.info(f"Start testing!")
    save_dir = os.path.join("../experiments", args.experiment, "codestream", "%02d" % (epoch + 1))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(
        net=net,
        test_dataloader=test_dataloader,
        logger_test=logger_test,
        save_dir=save_dir,
        epoch=epoch,
        mode=padding_mode,
    )


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
    main(sys.argv[1:])

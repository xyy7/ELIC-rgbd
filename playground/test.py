import os
import sys

import torch

sys.path.append("..")
import faulthandler

from config.args import test_options
from config.config import model_config
from PIL import Image, ImageFile
from testing.tester_master import TesterMaster
from testing.tester_single import TesterSingle
from testing.tester_united import TesterUnited
from testing.tester_concat import TesterConcat

faulthandler.enable()


def main(argv):
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options(argv)
    config = model_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.channel == 4:
        if args.model.find("cat") != -1:
            print("TesterConcat")
            tester = TesterConcat(args, config)
        else:
            print("TesterUnited")
            tester = TesterUnited(args, config)
    else:
        if args.model.find("master") != -1:
            print("TesterMaster")
            tester = TesterMaster(args, config)
        else:
            print("TesterSingle")
            tester = TesterSingle(args, config)
    tester.test_model(padding_mode="replicate0", padding=True)


if __name__ == "__main__":
    main(sys.argv[1:])

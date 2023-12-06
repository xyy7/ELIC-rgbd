import os
import sys

sys.path.append("..")  # cd xx/playground

import os

import torch
from utils.IOutils import saveImg
from utils.metrics import AverageMeter, compute_metrics

from .trainer import modelZoo
from .trainer_single import TrainerSingle


class TrainerMaster(TrainerSingle):
    def __init__(self, args, model_config) -> None:
        super().__init__(args, model_config)
        self.aux_net = modelZoo["ELIC"](config=model_config, channel=args.channel, return_mid=True).eval()
        self.ckpt_path1 = args.checkpoint1

    def forward(self, d):
        aux = d[0].to(self.device)
        d = d[1].to(self.device)
        with torch.no_grad():
            out = self.aux_net(aux)
        out_net = self.net(d, out["x_hat"], out)
        out_criterion = self.criterion(out_net, d)
        return out_criterion, out_net

    def restore(self, ckpt_path=None):
        epoch = super().restore(ckpt_path)
        aux_checkpoint = torch.load(self.ckpt_path1)
        self.aux_net.load_state_dict(aux_checkpoint["state_dict"])
        self.aux_net.update(force=True)
        self.aux_net = self.aux_net.to(self.device)
        return epoch


if __name__ == "__main__":
    pass

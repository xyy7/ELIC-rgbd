import torch
import torch.nn as nn
from compressai.layers import GDN, subpel_conv3x3
from modules.layers.conv import conv1x1, conv3x3


class ResidualBottleneck(nn.Module):

    def __init__(self, N=192, out=None, act=nn.ReLU) -> None:
        super().__init__()
        if out is None:
            out = N
        self.branch = nn.Sequential(
            conv1x1(N, N // 2), act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(), conv1x1(N // 2, out))
        if N != out:
            self.skip = conv1x1(N, out)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.branch(x)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out

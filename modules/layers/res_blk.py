import torch
import torch.nn as nn
from modules.layers.conv import conv1x1, conv3x3


class ResidualBottleneck(nn.Module):
    def __init__(self, N=192, out=None, act=nn.ReLU) -> None:
        super().__init__()

        out = N if out is not None else out
        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, out),
        )
        self.skip = conv1x1(N, out) if N != out else None

    def forward(self, x):
        identity = x
        out = self.branch(x)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out

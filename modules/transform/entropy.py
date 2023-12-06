import torch.nn as nn

from .attention import *


class EntropyParameters(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
        )

    def forward(self, params):
        """super().__init__()

        Args:
            params(Tensor): [B, C * K, H, W]
        return:
            gaussian_params(Tensor): [B, C * 2, H, W]
        """
        gaussian_params = self.fusion(params)

        return gaussian_params


class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 6, 1),
            act(),
            nn.Conv2d(in_dim // 6, out_dim * 4 // 3, kernel_size=3, stride=1, padding=1),  # 小范围的感受野有比较有帮助？
            act(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, kernel_size=5, stride=1, padding=2),
        )
        self.se = SE_Block(in_dim)  # 切换成通道

    def forward(self, params):
        """
        Args:
            params(Tensor): [B, C * K, H, W]
        return:
            gaussian_params(Tensor): [B, C * 2, H, W]
        """
        params = params + self.se(params)
        gaussian_params = self.fusion(params)

        return gaussian_params

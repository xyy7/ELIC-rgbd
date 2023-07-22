import torch.nn as nn
from compressai.layers import AttentionBlock
from modules.layers.conv import deconv
from modules.layers.res_blk import ResidualBottleneck

try:
    from analysis import *
except:
    from .analysis import *


class SynthesisTransformEX(nn.Module):
    def __init__(self, N, M, ch=3, act=nn.ReLU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, ch),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x


# 接受rgb和depth作为输入输出，但并没有进行交互
class SynthesisTransformEXcro(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.rgb_synthesis_transform = SynthesisTransformEX(N, M, ch=3)
        self.depth_synthesis_transform = SynthesisTransformEX(N, M, ch=1)

    def forward(self, rgb, depth):
        rgb = self.rgb_synthesis_transform(rgb)
        depth = self.depth_synthesis_transform(depth)
        return rgb, depth


class SynthesisTransformEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.rgb_synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            bi_spf(N),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            bi_spf(N),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            bi_spf(N),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 3),
        )
        # 使用identity进行占位
        self.depth_synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 1),
        )

    def forward(self, rgb, depth):
        rgb_hat = rgb
        depth_hat = depth
        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_synthesis_transform, self.depth_synthesis_transform)):
            if isinstance(rgb_bk, bi_spf):
                # depth_hat = depth_bk(depth_hat)
                rgb_f, depth_f = rgb_bk(rgb_hat, depth_hat)
                rgb_hat = torch.cat((rgb_hat, rgb_f), dim=-3)
                depth_hat = torch.cat((depth_hat, depth_f), dim=-3)
                # continue # 用于消融实验
            else:
                rgb_hat = rgb_bk(rgb_hat)
                depth_hat = depth_bk(depth_hat)

        return rgb_hat, depth_hat


class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.increase = nn.Sequential(deconv(N, M), act(), deconv(M, M * 3 // 2), act(), deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1))

    def forward(self, x):
        x = self.increase(x)
        return x


# 接受rgb和depth作为输入输出，不需要进行交互
class HyperSynthesisEXcro(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.rgb_increase = nn.Sequential(deconv(N, M), act(), deconv(M, M * 3 // 2), act(), deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1))
        self.depth_increase = nn.Sequential(deconv(N, M), act(), deconv(M, M * 3 // 2), act(), deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1))

    def forward(self, rgb, depth):
        rgb = self.rgb_increase(rgb)
        depth = self.depth_increase(depth)
        return rgb, depth


class HyperSynthesisEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.r_h_s1 = hyper_transform_block(2 * N, M)
        self.r_h_s2 = hyper_transform_block(2 * M, M * 3 // 2)
        self.r_h_s3 = hyper_transform_block(M * 3, 2 * M, True)

        self.d_h_s1 = hyper_transform_block(2 * N, M)
        self.d_h_s2 = hyper_transform_block(2 * M, M * 3 // 2)
        self.d_h_s3 = hyper_transform_block(M * 3, 2 * M, True)

    def forward(self, rgb, depth):
        r1 = self.r_h_s1(rgb, depth)
        d1 = self.d_h_s1(depth, rgb)
        r2 = self.r_h_s2(r1, d1)
        d2 = self.d_h_s2(d1, r1)
        r_params = self.r_h_s3(r2, d2)
        d_params = self.d_h_s3(d2, r2)
        return r_params, d_params


class hyper_transform_block(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(hyper_transform_block, self).__init__()
        self.se = SE_Block(in_channel)
        if is_last is False:
            self.relu = nn.LeakyReLU(inplace=True)
            self.deconv = deconv(in_channel, out_channel, stride=2, kernel_size=5)
        else:
            self.relu = None
            self.deconv = deconv(in_channel, out_channel, stride=1, kernel_size=3)  # united 发生了改变

    def forward(self, rgb, depth):
        f = torch.cat((rgb, depth), dim=-3)
        f = self.se(f)
        f = self.deconv(f)
        if self.relu is not None:
            f = self.relu(f)
        return f


# rgb和depth根据空间均值来进行通道加权【通过全局池化来实现】
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(nn.Linear(ch_in, ch_in // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(ch_in // reduction, ch_in, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道

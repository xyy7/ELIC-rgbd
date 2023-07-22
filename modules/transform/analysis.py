import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import AttentionBlock, subpel_conv3x3
from modules.layers.conv import conv, conv1x1, conv3x3, deconv
from modules.layers.res_blk import *


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f = x
        c1_ = self.conv1(f)  # 1*1卷积，降低维度（减少计算复杂度）
        c1 = self.conv2(c1_)  # 减小特征图尺寸
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # 减小特征图尺寸，增大感受野
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)  # 上采样，恢复特征图尺寸
        cf = self.conv_f(c1_)  #
        c4 = self.conv4(c3 + cf)  # 1*1卷积恢复通道数
        m = self.sigmoid(c4)  # 生成mask

        return x * m


# 去掉fusion，ext output channel减半
# 命名为ESA，反而使用的是空间注意力？
class bi_spf(nn.Module):
    def __init__(self, N):
        super(bi_spf, self).__init__()
        self.r_ext = nn.Conv2d(N, N // 2, stride=1, kernel_size=3, padding=1)
        self.r_act = nn.ReLU()
        self.r_esa = ESA(N)

        self.d_ext = nn.Conv2d(N, N // 2, stride=1, kernel_size=3, padding=1)
        self.d_act = nn.ReLU()
        self.d_esa = ESA(N)

    def forward(self, rgb, depth):
        rgb = self.r_ext(rgb)
        rgb = self.r_act(rgb)
        depth = self.d_ext(depth)
        depth = self.d_act(depth)

        r = self.r_esa(torch.cat((rgb, depth), dim=-3))
        d = self.d_esa(torch.cat((depth, rgb), dim=-3))

        return r, d


class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, ch=3, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(ch, N),  # 通过卷积进行下采样
            ResidualBottleneck(N, act=act),  # 通过残差块来增强特征
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, M),
            AttentionBlock(M),  # 通过通道注意力，进一步增强特征
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x


# 接受rgb和depth作为输入输出，但并没有进行交互
class AnalysisTransformEXcro(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.rgb_analysis_transform = AnalysisTransformEX(N, M, ch=3)
        self.depth_analysis_transform = AnalysisTransformEX(N, M, ch=1)

    def forward(self, rgb, depth):
        rgb = self.rgb_analysis_transform(rgb)
        depth = self.depth_analysis_transform(depth)
        return rgb, depth


# 接受rgb和depth作为输入输出，但需要进行交互
class AnalysisTransformEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.rgb_analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            bi_spf(N),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            bi_spf(N),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            bi_spf(N),
            conv(2 * N, M),
            AttentionBlock(M),
        )

        self.depth_analysis_transform = nn.Sequential(
            conv(1, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            nn.Identity(),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            nn.Identity(),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            nn.Identity(),
            conv(2 * N, M),
            AttentionBlock(M),
        )

    def forward(self, rgb, depth):
        rgb_y = rgb
        depth_y = depth
        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_analysis_transform, self.depth_analysis_transform)):
            if isinstance(rgb_bk, bi_spf):
                # depth_y = depth_bk(depth_y)
                rgb_f, depth_f = rgb_bk(rgb_y, depth_y)
                rgb_y = torch.cat((rgb_y, rgb_f), dim=-3)
                depth_y = torch.cat((depth_y, depth_f), dim=-3)
                # continue # 用于消融实验
            else:
                rgb_y = rgb_bk(rgb_y)
                depth_y = depth_bk(depth_y)

        return rgb_y, depth_y


class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))

    def forward(self, x):
        x = self.reduction(x)
        return x


# 接受rgb和depth作为输入输出，但并没有进行交互
class HyperAnalysisEXcro(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.rgb_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))
        self.depth_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))

    def forward(self, rgb, depth):
        rgb = self.rgb_reduction(rgb)
        depth = self.depth_reduction(depth)
        return rgb, depth


# 不需要重写，因为本来就没有改进
class HyperAnalysisEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.rgb_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))
        self.depth_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))

    def forward(self, rgb, depth):
        rgb = self.rgb_reduction(rgb)
        depth = self.depth_reduction(depth)
        return rgb, depth


class HyperAnalysisEXcross3(nn.Module):
    def __init__(self, M, N, act=nn.ReLU) -> None:
        super().__init__()
        self.r_h_s1 = hyper_transform_block(2 * N, M)
        self.r_h_s2 = hyper_transform_block(2 * M, M * 3 // 2)
        self.r_h_s3 = hyper_transform_block(M * 3, M, True)

        self.d_h_s1 = hyper_transform_block(2 * N, M)
        self.d_h_s2 = hyper_transform_block(2 * M, M * 3 // 2)
        self.d_h_s3 = hyper_transform_block(M * 3, M, True)

    def forward(self, rgb, depth):
        r1 = self.r_h_s1(rgb, depth)
        d1 = self.d_h_s1(depth, rgb)
        r2 = self.r_h_s2(r1, d1)
        d2 = self.d_h_s2(d1, r1)
        rgb = self.r_h_s3(r2, d2)
        depth = self.d_h_s3(d2, r2)
        return rgb, depth


class hyper_transform_block(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(hyper_transform_block, self).__init__()
        self.se = SE_Block(in_channel)
        if is_last is False:
            self.relu = nn.LeakyReLU(inplace=True)
            self.conv = conv(in_channel, out_channel)
        else:
            self.relu = None
            self.conv = conv3x3(in_channel, out_channel)

    def forward(self, rgb, depth):
        f = torch.cat((rgb, depth), dim=-3)
        f = self.se(f)
        f = self.conv(f)
        if self.relu is not None:
            f = self.relu(f)
        return f


# rgb和depth通道加权【通过全局池化来实现】
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

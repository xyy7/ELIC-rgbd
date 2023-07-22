import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.ops import ste_round
from modules.transform import *
from utils.ckbd import *
from utils.func import get_scale_table, update_registered_buffers


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
        tmp = params
        params = self.se(params)  # 还结合了空间信息
        params += tmp
        gaussian_params = self.fusion(params)

        return gaussian_params


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


class EntropyParametersEX1(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 6, 1),
            act(),
            nn.Conv2d(in_dim // 6, out_dim * 4 // 3, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, kernel_size=3, stride=1, padding=1),
        )
        self.se = SE_Block_plus(in_dim)  # 切换成通道

    def forward(self, params):
        """
        Args:
            params(Tensor): [B, C * K, H, W]
        return:
            gaussian_params(Tensor): [B, C * 2, H, W]
        """
        shortcut = params
        params = self.se(params)  # 还结合了空间信息
        params += shortcut
        gaussian_params = self.fusion(params)

        return gaussian_params


class spatial_attn1(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.rgb_conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.depth_conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.rgb_conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.depth_conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, depth):
        output = rgb
        p = torch.tanh(self.rgb_conv1(rgb) + self.depth_conv1(depth))
        s = torch.sigmoid(self.rgb_conv2(rgb) + self.depth_conv2(depth))
        return s * p + (1 - s) * output


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


class SE_Block_plus(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block_plus, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(nn.Linear(ch_in, ch_in // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(ch_in // reduction, ch_in, bias=False), nn.Sigmoid())

        self.x_conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // reduction, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, ch_in // reduction // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction // 2, ch_in // reduction, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, ch_in, 1, 1),
        )
        self.x_ch_conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // reduction, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, ch_in // reduction // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction // 2, ch_in // reduction, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, ch_in, 1, 1),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        x_ch = x * y.expand_as(x)
        s = torch.sigmoid(self.x_conv1(x) + self.x_ch_conv1(x_ch))  # 再添加一个空间注意力
        return s * x_ch + (1 - s) * x


class ELIC_united_EEM(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch  # [8, 8, 8, 8, 16, 16, 32, 32, 96, 96]
        self.quant = config.quant  # noise or ste
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEXcro(N, M, act=nn.ReLU)  # 只是使用EEM
        self.g_s = SynthesisTransformEXcro(N, M, act=nn.ReLU)
        # Hyper Transform【因子先验的超先ss验加强+1】
        self.h_a = HyperAnalysisEXcross(N, M, act=nn.ReLU)  # 355 122
        self.h_s = HyperSynthesisEXcross(N, M, act=nn.ReLU)

        # 这里改成了两倍，进行local跨模态交互
        # Channel Fusion Model
        # self.rgb_local_context = nn.ModuleList(
        #     nn.Conv2d(in_channels=slice_ch[i] * 2, out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
        #     if i
        #     else nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
        #     for i in range(len(slice_ch))
        # )
        # self.depth_local_context = nn.ModuleList(
        #     nn.Conv2d(in_channels=slice_ch[i] * 2, out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
        #     if i
        #     else nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
        #     for i in range(len(slice_ch))
        # )
        self.rgb_local_context = nn.ModuleList(nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2) for i in range(len(slice_ch)))
        self.rgb_local_context_anchor_with_nonanchor = nn.ModuleList(nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2) for i in range(len(slice_ch)))
        self.depth_local_context = nn.ModuleList(nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2) for i in range(len(slice_ch)))

        # 因为是使用索引idx来跳过None，第一个idx，没有使用channel信息
        self.rgb_channel_context = nn.ModuleList(ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None for i in range(slice_num))
        self.depth_channel_context = nn.ModuleList(ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None for i in range(slice_num))

        # Use channel_ctx and hyper_params 【+ cross modal channel ctx + cross modal hyper_params + cross modal local for depth】
        # 在预测param的时候，再添加另外一个模态的信息，不要让原来的信息被挤掉
        # self.rgb_entropy_parameters_anchor = nn.ModuleList(
        #     EntropyParametersEX(in_dim=(M * 2 + slice_ch[i] * 2), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
        #     for i in range(slice_num)
        # )
        # self.depth_entropy_parameters_anchor = nn.ModuleList(
        #     EntropyParametersEX(in_dim=(M * 2 + slice_ch[i] * 2), out_dim=slice_ch[i] * 2, act=nn.ReLU)
        #     if i
        #     else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)  # 首个不使用channel信息
        #     for i in range(slice_num)
        # )
        # 现在先堆材料
        self.rgb_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=(M * 4 + slice_ch[i] * 4), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else EntropyParametersEX(in_dim=M * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        # depth可以使用rgb解码完的local-anchor信息
        self.depth_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=(M * 4 + slice_ch[i] * 6), out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        # Use channel_ctx and hyper_params 【+ cross modal channel ctx + cross modal hyper_params + cross modal local for rgb + depth】
        # rgb可以使用rgb解码完和depth的local-anchor信息
        self.rgb_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        # depth可以使用rgb解码完和depth的local-anchor信息，以及rgb解码完的local-nonanchor
        self.depth_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # 因子先验
        self.entropy_bottleneck = None
        self.rgb_entropy_bottleneck = EntropyBottleneck(N)
        self.depth_entropy_bottleneck = EntropyBottleneck(N)

        # Gussian Conditional
        self.gaussianConditional = None
        self.rgb_gaussian_conditional = GaussianConditional(None)
        self.depth_gaussian_conditional = GaussianConditional(None)

    # def entropy_estimate_index0(self, y_slice, hyper_params, gaussian_conditional, local_context, idx, entropy_parameters_anchor, entropy_parameters_nonanchor):
    #     slice_anchor, slice_nonanchor = ckbd_split(y_slice)
    #     # Anchor
    #     params_anchor = entropy_parameters_anchor[idx](hyper_params)
    #     scales_anchor, means_anchor = params_anchor.chunk(2, 1)
    #     # split means and scales of anchor
    #     scales_anchor = ckbd_anchor(scales_anchor)
    #     means_anchor = ckbd_anchor(means_anchor)
    #     # round anchor
    #     if self.quant == "ste":
    #         slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
    #     else:
    #         slice_anchor = gaussian_conditional.quantize(slice_anchor, "noise" if self.training else "dequantize")
    #         slice_anchor = ckbd_anchor(slice_anchor)
    #     # Non-anchor
    #     # local_ctx: [B, H, W, 2 * C]
    #     local_ctx = local_context[idx](slice_anchor)
    #     params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))  # 这里其实也是可以考虑交互的
    #     scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
    #     # split means and scales of nonanchor
    #     scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
    #     means_nonanchor = ckbd_nonanchor(means_nonanchor)
    #     # merge means and scales of anchor and nonanchor
    #     scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
    #     means_slice = ckbd_merge(means_anchor, means_nonanchor)
    #     _, y_slice_likelihoods = gaussian_conditional(y_slice, scales_slice, means_slice)
    #     # round slice_nonanchor
    #     if self.quant == "ste":
    #         slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
    #     else:
    #         slice_nonanchor = gaussian_conditional.quantize(slice_nonanchor, "noise" if self.training else "dequantize")
    #         slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
    #     y_hat_slice = slice_anchor + slice_nonanchor
    #     return y_hat_slice, y_slice_likelihoods

    def entropy_estimate_index0(self, rgb_y_slice, rgb_hyper_params, depth_y_slice, depth_hyper_params):
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)
        # rgb_Anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[0](torch.cat([rgb_hyper_params, depth_hyper_params], dim=1))
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        # split means and scales of anchor
        rgb_scales_anchor = ckbd_anchor(rgb_scales_anchor)
        rgb_means_anchor = ckbd_anchor(rgb_means_anchor)
        # round anchor
        if self.quant == "ste":
            rgb_slice_anchor = ste_round(rgb_slice_anchor - rgb_means_anchor) + rgb_means_anchor
        else:
            rgb_slice_anchor = self.rgb_gaussian_conditional.quantize(rgb_slice_anchor, "noise" if self.training else "dequantize")
            rgb_slice_anchor = ckbd_anchor(rgb_slice_anchor)
        rgb_local_ctx = self.rgb_local_context[0](rgb_slice_anchor)

        # depth_Anchor
        depth_params_anchor = self.depth_entropy_parameters_anchor[0](torch.cat([depth_hyper_params, depth_hyper_params, rgb_local_ctx], dim=1))
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        # split means and scales of anchor
        depth_scales_anchor = ckbd_anchor(depth_scales_anchor)
        depth_means_anchor = ckbd_anchor(depth_means_anchor)
        # round anchor
        if self.quant == "ste":
            depth_slice_anchor = ste_round(depth_slice_anchor - depth_means_anchor) + depth_means_anchor
        else:
            depth_slice_anchor = self.depth_gaussian_conditional.quantize(depth_slice_anchor, "noise" if self.training else "dequantize")
            depth_slice_anchor = ckbd_anchor(depth_slice_anchor)
        depth_local_ctx = self.depth_local_context[0](depth_slice_anchor)

        # rgb_Non-anchor
        # local_ctx: [B, H, W, 2 * C]
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[0](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里其实也是可以考虑交互的
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        # split means and scales of nonanchor
        rgb_scales_nonanchor = ckbd_nonanchor(rgb_scales_nonanchor)
        rgb_means_nonanchor = ckbd_nonanchor(rgb_means_nonanchor)
        # merge means and scales of anchor and nonanchor
        rgb_scales_slice = ckbd_merge(rgb_scales_anchor, rgb_scales_nonanchor)
        rgb_means_slice = ckbd_merge(rgb_means_anchor, rgb_means_nonanchor)
        _, rgb_y_slice_likelihoods = self.rgb_gaussian_conditional(rgb_y_slice, rgb_scales_slice, rgb_means_slice)
        # round slice_nonanchor
        if self.quant == "ste":
            rgb_slice_nonanchor = ste_round(rgb_slice_nonanchor - rgb_means_nonanchor) + rgb_means_nonanchor
        else:
            rgb_slice_nonanchor = self.rgb_gaussian_conditional.quantize(rgb_slice_nonanchor, "noise" if self.training else "dequantize")
            rgb_slice_nonanchor = ckbd_nonanchor(rgb_slice_nonanchor)

        rgb_y_hat_slice = rgb_slice_anchor + rgb_slice_nonanchor
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[0](rgb_y_hat_slice)

        # depth_Non-anchor
        # local_ctx: [B, H, W, 2 * C]
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[0](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里其实也是可以考虑交互的
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        # split means and scales of nonanchor
        depth_scales_nonanchor = ckbd_nonanchor(depth_scales_nonanchor)
        depth_means_nonanchor = ckbd_nonanchor(depth_means_nonanchor)
        # merge means and scales of anchor and nonanchor
        depth_scales_slice = ckbd_merge(depth_scales_anchor, depth_scales_nonanchor)
        depth_means_slice = ckbd_merge(depth_means_anchor, depth_means_nonanchor)
        _, depth_y_slice_likelihoods = self.depth_gaussian_conditional(depth_y_slice, depth_scales_slice, depth_means_slice)
        # round slice_nonanchor
        if self.quant == "ste":
            depth_slice_nonanchor = ste_round(depth_slice_nonanchor - depth_means_nonanchor) + depth_means_nonanchor
        else:
            depth_slice_nonanchor = self.depth_gaussian_conditional.quantize(depth_slice_nonanchor, "noise" if self.training else "dequantize")
            depth_slice_nonanchor = ckbd_nonanchor(depth_slice_nonanchor)

        depth_y_hat_slice = depth_slice_anchor + depth_slice_nonanchor

        return rgb_y_hat_slice, rgb_y_slice_likelihoods, depth_y_hat_slice, depth_y_slice_likelihoods

    def get_anchor(self, slice_anchor, params_anchor, gaussian_conditional):
        scales_anchor, means_anchor = params_anchor.chunk(2, 1)
        # split means and scales of anchor
        scales_anchor = ckbd_anchor(scales_anchor)  # 函数覆盖了，吃了没有深拷贝的亏
        means_anchor = ckbd_anchor(means_anchor)
        # round anchor
        if self.quant == "ste":
            slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
        else:
            slice_anchor = gaussian_conditional.quantize(slice_anchor, "noise" if self.training else "dequantize")
            slice_anchor = ckbd_anchor(slice_anchor)
        return slice_anchor

    def get_nonanchor(self, params_nonanchor, slice_nonanchor, gaussian_conditional):
        # scales_anchor, means_anchor = params_anchor.chunk(2, 1)
        scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
        # split means and scales of nonanchor
        scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
        means_nonanchor = ckbd_nonanchor(means_nonanchor)
        if self.quant == "ste":
            slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
        else:
            slice_nonanchor = gaussian_conditional.quantize(slice_nonanchor, "noise" if self.training else "dequantize")
            slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
        return slice_nonanchor

    def get_total_y_likelihood(self, params_anchor, params_nonanchor, y_slice, gaussian_conditional):
        scales_anchor, means_anchor = params_anchor.chunk(2, 1)
        scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
        # 写代码的时候少了这一行，导致forward的时候没问题，但是在compress的时候就出现问题
        # split means and scales of nonanchor
        scales_anchor = ckbd_anchor(scales_anchor)
        means_anchor = ckbd_anchor(means_anchor)

        scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
        means_nonanchor = ckbd_nonanchor(means_nonanchor)
        # merge means and scales of anchor and nonanchor
        scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
        means_slice = ckbd_merge(means_anchor, means_nonanchor)

        _, y_slice_likelihoods = gaussian_conditional(y_slice, scales_slice, means_slice)
        return y_slice_likelihoods

    def entropy_estimate_index1234(self, rgb_y_slice, depth_y_slice, rgb_hyper_params, depth_hyper_params, rgb_y_hat_slices, depth_y_hat_slices, idx):
        # 先验知识：hyper+ch-cross
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)
        # # ch-cross
        # rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices + depth_y_hat_slices, dim=1))  # 如果是这样作为上下文，有可能出现把原来的信息给挤掉了，因为channel数量没有改变
        # depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices + rgb_y_hat_slices, dim=1))

        rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))  # 如果是这样作为上下文，有可能出现把原来的信息给挤掉了，因为channel数量没有改变
        depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))

        # Anchor(Use channel context and hyper params)
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](torch.cat([rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里仍然可以添加另一个模态的信息
        rgb_slice_anchor = self.get_anchor(rgb_slice_anchor, rgb_params_anchor, self.rgb_gaussian_conditional)
        rgb_local_ctx = self.rgb_local_context[idx](rgb_slice_anchor)

        depth_params_anchor = self.depth_entropy_parameters_anchor[idx](torch.cat([rgb_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))
        depth_slice_anchor = self.get_anchor(depth_slice_anchor, depth_params_anchor, self.depth_gaussian_conditional)
        depth_local_ctx = self.depth_local_context[idx](depth_slice_anchor)

        # 先验知识：hyper+local
        # ctx_params: [B, H, W, 2 * C]
        # local-cross
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[idx](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))
        rgb_slice_nonanchor = self.get_nonanchor(rgb_params_nonanchor, rgb_slice_nonanchor, self.rgb_gaussian_conditional)
        rgb_y_slice_likelihoods = self.get_total_y_likelihood(rgb_params_anchor, rgb_params_nonanchor, rgb_y_slice, self.rgb_gaussian_conditional)
        rgb_y_hat_slice = rgb_slice_anchor + rgb_slice_nonanchor
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)

        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx, depth_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1)
        )
        depth_slice_nonanchor = self.get_nonanchor(depth_params_nonanchor, depth_slice_nonanchor, self.depth_gaussian_conditional)
        depth_y_slice_likelihoods = self.get_total_y_likelihood(depth_params_anchor, depth_params_nonanchor, depth_y_slice, self.depth_gaussian_conditional)

        depth_y_hat_slice = depth_slice_anchor + depth_slice_nonanchor
        return rgb_y_hat_slice, depth_y_hat_slice, rgb_y_slice_likelihoods, depth_y_slice_likelihoods

    def entropy_estimate_united(self, rgb, depth, rgb_hyper_params, depth_hyper_params):
        rgb_y_slices = [rgb[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))]
        rgb_y_hat_slices = []
        rgb_y_likelihoods = []

        depth_y_slices = [depth[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))]
        depth_y_hat_slices = []
        depth_y_likelihoods = []

        for idx, (rgb_y_slice, depth_y_slice) in enumerate(zip(rgb_y_slices, depth_y_slices)):
            if idx == 0:
                rgb_y_hat_slice, rgb_y_slice_likelihoods, depth_y_hat_slice, depth_y_slice_likelihoods = self.entropy_estimate_index0(rgb_y_slice, rgb_hyper_params, depth_y_slice, depth_hyper_params)

                rgb_y_hat_slices.append(rgb_y_hat_slice)
                rgb_y_likelihoods.append(rgb_y_slice_likelihoods)

                depth_y_hat_slices.append(depth_y_hat_slice)
                depth_y_likelihoods.append(depth_y_slice_likelihoods)

            else:
                rgb_y_hat_slice, depth_y_hat_slice, rgb_y_slice_likelihoods, depth_y_slice_likelihoods = self.entropy_estimate_index1234(
                    rgb_y_slice, depth_y_slice, rgb_hyper_params, depth_hyper_params, rgb_y_hat_slices, depth_y_hat_slices, idx
                )
                rgb_y_hat_slices.append(rgb_y_hat_slice)
                rgb_y_likelihoods.append(rgb_y_slice_likelihoods)
                depth_y_hat_slices.append(depth_y_hat_slice)
                depth_y_likelihoods.append(depth_y_slice_likelihoods)

        rgb_y_hat = torch.cat(rgb_y_hat_slices, dim=1)
        rgb_y_likelihoods = torch.cat(rgb_y_likelihoods, dim=1)

        depth_y_hat = torch.cat(depth_y_hat_slices, dim=1)
        depth_y_likelihoods = torch.cat(depth_y_likelihoods, dim=1)

        return rgb_y_hat, rgb_y_likelihoods, depth_y_hat, depth_y_likelihoods

    def forward(self, rgb, depth):
        rgb_y, depth_y = self.g_a(rgb, depth)
        rgb_z, depth_z = self.h_a(rgb_y, depth_y)

        # bits先验估计
        rgb_z_hat, rgb_z_likelihoods = self.rgb_entropy_bottleneck(rgb_z)
        depth_z_hat, depth_z_likelihoods = self.depth_entropy_bottleneck(depth_z)
        if self.quant == "ste":
            rgb_z_offset = self.rgb_entropy_bottleneck._get_medians()
            rgb_z_hat = ste_round(rgb_z - rgb_z_offset) + rgb_z_offset
            depth_z_offset = self.depth_entropy_bottleneck._get_medians()
            depth_z_hat = ste_round(depth_z - depth_z_offset) + depth_z_offset

        # Hyper-parameters
        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)
        rgb_y_hat, rgb_y_likelihoods, depth_y_hat, depth_y_likelihoods = self.entropy_estimate_united(rgb_y, depth_y, rgb_hyper_params, depth_hyper_params)

        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        return {"x_hat": {"r": rgb_hat, "d": depth_hat}, "r_likelihoods": {"y": rgb_y_likelihoods, "z": rgb_z_likelihoods}, "d_likelihoods": {"y": depth_y_likelihoods, "z": depth_z_likelihoods}}

    # def compress_index0(self, y_slice, hyper_params, gaussian_conditional, local_context, idx, entropy_parameters_anchor, entropy_parameters_nonanchor, symbols_list, indexes_list):
    #     # Anchor
    #     slice_anchor, slice_nonanchor = ckbd_split(y_slice)

    #     params_anchor = entropy_parameters_anchor[idx](hyper_params)
    #     scales_anchor, means_anchor = params_anchor.chunk(2, 1)
    #     # round and compress anchor
    #     slice_anchor = compress_anchor(gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
    #     # Non-anchor
    #     # local_ctx: [B,2 * C, H, W]
    #     local_ctx = local_context[idx](slice_anchor)
    #     params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
    #     scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
    #     # round and compress nonanchor
    #     slice_nonanchor = compress_nonanchor(gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
    #     y_slice_hat = slice_anchor + slice_nonanchor
    #     return y_slice_hat
    def compress_index0(self, rgb_y_slice, rgb_hyper_params, rgb_symbols_list, rgb_indexes_list, depth_y_slice, depth_hyper_params, depth_symbols_list, depth_indexes_list):
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)

        # rgb-Anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[0](torch.cat([rgb_hyper_params, depth_hyper_params], dim=1))
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        # round and compress anchor
        rgb_slice_anchor = compress_anchor(self.rgb_gaussian_conditional, rgb_slice_anchor, rgb_scales_anchor, rgb_means_anchor, rgb_symbols_list, rgb_indexes_list)
        # Non-anchor
        # local_ctx: [B,2 * C, H, W]
        rgb_local_ctx = self.rgb_local_context[0](rgb_slice_anchor)

        # depth-anchor
        depth_params_anchor = self.depth_entropy_parameters_anchor[0](torch.cat([depth_hyper_params, depth_hyper_params, rgb_local_ctx], dim=1))
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        depth_slice_anchor = compress_anchor(self.depth_gaussian_conditional, depth_slice_anchor, depth_scales_anchor, depth_means_anchor, depth_symbols_list, depth_indexes_list)
        depth_local_ctx = self.depth_local_context[0](depth_slice_anchor)

        # rgb-nonanchor
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[0](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里其实也是可以考虑交互的
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        # round and compress nonanchor
        rgb_slice_nonanchor = compress_nonanchor(self.rgb_gaussian_conditional, rgb_slice_nonanchor, rgb_scales_nonanchor, rgb_means_nonanchor, rgb_symbols_list, rgb_indexes_list)
        rgb_y_hat_slice = rgb_slice_anchor + rgb_slice_nonanchor
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[0](rgb_y_hat_slice)

        # depth-nonanchor
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[0](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里其实也是可以考虑交互的
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        # split means and scales of nonanchor
        depth_scales_nonanchor = ckbd_nonanchor(depth_scales_nonanchor)
        depth_means_nonanchor = ckbd_nonanchor(depth_means_nonanchor)
        depth_slice_nonanchor = compress_nonanchor(self.depth_gaussian_conditional, depth_slice_nonanchor, depth_scales_nonanchor, depth_means_nonanchor, depth_symbols_list, depth_indexes_list)
        depth_y_hat_slice = depth_slice_anchor + depth_slice_nonanchor

        return rgb_y_hat_slice, depth_y_hat_slice

    def compress_index1234(
        self, rgb_y_slice, depth_y_slice, rgb_hyper_params, depth_hyper_params, rgb_y_hat_slices, depth_y_hat_slices, idx, rgb_symbols_list, rgb_indexes_list, depth_symbols_list, depth_indexes_list
    ):
        # 先验知识：hyper+ch-cross
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)
        # ch-cross
        rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))  # 如果是这样作为上下文，有可能出现把原来的信息给挤掉了，因为channel数量没有改变
        depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))

        # rgb-anchor
        # Anchor(Use channel context and hyper params)
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](torch.cat([rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        rgb_slice_anchor = compress_anchor(self.rgb_gaussian_conditional, rgb_slice_anchor, rgb_scales_anchor, rgb_means_anchor, rgb_symbols_list, rgb_indexes_list)
        rgb_local_ctx = self.rgb_local_context[idx](rgb_slice_anchor)

        # depth-anchor
        depth_params_anchor = self.depth_entropy_parameters_anchor[idx](torch.cat([rgb_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        depth_slice_anchor = compress_anchor(self.depth_gaussian_conditional, depth_slice_anchor, depth_scales_anchor, depth_means_anchor, depth_symbols_list, depth_indexes_list)
        depth_local_ctx = self.depth_local_context[idx](depth_slice_anchor)

        # rgb-nonanchor
        # local_ctx: [B,2 * C, H, W]
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx, depth_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1)
        )  # 这里两个local是有问题的

        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        # round and compress nonanchor
        rgb_slice_nonanchor = compress_nonanchor(self.rgb_gaussian_conditional, rgb_slice_nonanchor, rgb_scales_nonanchor, rgb_means_nonanchor, rgb_symbols_list, rgb_indexes_list)
        rgb_y_hat_slice = rgb_slice_nonanchor + rgb_slice_anchor
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)

        # depth-nonanchor
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx, depth_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1)
        )
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        depth_slice_nonanchor = compress_nonanchor(self.depth_gaussian_conditional, depth_slice_nonanchor, depth_scales_nonanchor, depth_means_nonanchor, depth_symbols_list, depth_indexes_list)
        depth_y_hat_slice = depth_slice_nonanchor + depth_slice_anchor

        rgb_y_hat_slices.append(rgb_y_hat_slice)
        depth_y_hat_slices.append(depth_y_hat_slice)
        return rgb_y_hat_slices, depth_y_hat_slices

    def feature2bin_united(self, rgb_y, rgb_hyper_params, depth_y, depth_hyper_params):
        rgb_y_slices = [rgb_y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))]
        rgb_y_hat_slices = []

        depth_y_slices = [depth_y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))]
        depth_y_hat_slices = []

        rgb_cdf = self.rgb_gaussian_conditional.quantized_cdf.tolist()
        rgb_cdf_lengths = self.rgb_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        rgb_offsets = self.rgb_gaussian_conditional.offset.reshape(-1).int().tolist()
        rgb_encoder = BufferedRansEncoder()
        rgb_symbols_list = []
        rgb_indexes_list = []
        rgb_y_strings = []

        depth_cdf = self.depth_gaussian_conditional.quantized_cdf.tolist()
        depth_cdf_lengths = self.depth_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        depth_offsets = self.depth_gaussian_conditional.offset.reshape(-1).int().tolist()
        depth_encoder = BufferedRansEncoder()
        depth_symbols_list = []
        depth_indexes_list = []
        depth_y_strings = []

        for idx, (rgb_y_slice, depth_y_slice) in enumerate(zip(rgb_y_slices, depth_y_slices)):
            if idx == 0:
                rgb_y_hat_slice, depth_y_hat_slice = self.compress_index0(
                    rgb_y_slice, rgb_hyper_params, rgb_symbols_list, rgb_indexes_list, depth_y_slice, depth_hyper_params, depth_symbols_list, depth_indexes_list
                )
                rgb_y_hat_slices.append(rgb_y_hat_slice)

                depth_y_hat_slices.append(depth_y_hat_slice)

            else:
                rgb_y_hat_slices, depth_y_hat_slices = self.compress_index1234(
                    rgb_y_slice,
                    depth_y_slice,
                    rgb_hyper_params,
                    depth_hyper_params,
                    rgb_y_hat_slices,
                    depth_y_hat_slices,
                    idx,
                    rgb_symbols_list,
                    rgb_indexes_list,
                    depth_symbols_list,
                    depth_indexes_list,
                )

        rgb_encoder.encode_with_indexes(rgb_symbols_list, rgb_indexes_list, rgb_cdf, rgb_cdf_lengths, rgb_offsets)
        rgb_y_string = rgb_encoder.flush()
        rgb_y_strings.append(rgb_y_string)

        depth_encoder.encode_with_indexes(depth_symbols_list, depth_indexes_list, depth_cdf, depth_cdf_lengths, depth_offsets)
        depth_y_string = depth_encoder.flush()
        depth_y_strings.append(depth_y_string)
        return rgb_y_strings, depth_y_strings

    def compress(self, rgb, depth):
        rgb_y, depth_y = self.g_a(rgb, depth)
        rgb_z, depth_z = self.h_a(rgb_y, depth_y)

        # bits先验
        torch.backends.cudnn.deterministic = True
        rgb_z_strings = self.rgb_entropy_bottleneck.compress(rgb_z)
        rgb_z_hat = self.rgb_entropy_bottleneck.decompress(rgb_z_strings, rgb_z.size()[-2:])
        depth_z_strings = self.depth_entropy_bottleneck.compress(depth_z)
        depth_z_hat = self.depth_entropy_bottleneck.decompress(depth_z_strings, depth_z.size()[-2:])

        # Hyper-parameters
        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)
        rgb_y_strings, depth_y_strings = self.feature2bin_united(rgb_y, rgb_hyper_params, depth_y, depth_hyper_params)

        torch.backends.cudnn.deterministic = False
        return {"r_strings": [rgb_y_strings, rgb_z_strings], "d_strings": [depth_y_strings, depth_z_strings], "shape": rgb_z.size()[-2:]}

    def decompress(self, rgb_strings, depth_strings, shape):
        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()
        start_time = time.process_time()

        rgb_y_strings = rgb_strings[0][0]  # 本来不需要的，只是compress写成了列表的形式
        rgb_z_strings = rgb_strings[1]
        rgb_z_hat = self.rgb_entropy_bottleneck.decompress(rgb_z_strings, shape)
        depth_y_strings = depth_strings[0][0]
        depth_z_strings = depth_strings[1]
        depth_z_hat = self.depth_entropy_bottleneck.decompress(depth_z_strings, shape)

        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)

        rgb_y_hat, depth_y_hat = self.bin2img_united(rgb_y_strings, rgb_hyper_params, depth_y_strings, depth_hyper_params)
        torch.backends.cudnn.deterministic = False
        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time
        return {"x_hat": {"r": rgb_hat.clamp_(0, 1), "d": depth_hat.clamp_(0, 1)}, "cost_time": cost_time}

    def decompress_index0(self, rgb_decoder, rgb_hyper_params, rgb_cdf, rgb_cdf_lengths, rgb_offsets, depth_decoder, depth_hyper_params, depth_cdf, depth_cdf_lengths, depth_offsets):
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[0](torch.cat([rgb_hyper_params, depth_hyper_params], dim=1))
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        # decompress anchor
        rgb_slice_anchor = decompress_anchor(self.rgb_gaussian_conditional, rgb_scales_anchor, rgb_means_anchor, rgb_decoder, rgb_cdf, rgb_cdf_lengths, rgb_offsets)
        # Non-anchor
        # local_ctx: [B,2 * C, H, W]
        rgb_local_ctx = self.rgb_local_context[0](rgb_slice_anchor)

        depth_params_anchor = self.depth_entropy_parameters_anchor[0](torch.cat([depth_hyper_params, depth_hyper_params, rgb_local_ctx], dim=1))
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        # decompress anchor
        depth_slice_anchor = decompress_anchor(self.depth_gaussian_conditional, depth_scales_anchor, depth_means_anchor, depth_decoder, depth_cdf, depth_cdf_lengths, depth_offsets)
        # Non-anchor
        # local_ctx: [B,2 * C, H, W]
        depth_local_ctx = self.depth_local_context[0](depth_slice_anchor)

        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[0](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里其实也是可以考虑交互的
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        # decompress non-anchor
        rgb_slice_nonanchor = decompress_nonanchor(self.rgb_gaussian_conditional, rgb_scales_nonanchor, rgb_means_nonanchor, rgb_decoder, rgb_cdf, rgb_cdf_lengths, rgb_offsets)
        rgb_y_hat_slice = rgb_slice_nonanchor + rgb_slice_anchor
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[0](rgb_y_hat_slice)

        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[0](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里其实也是可以考虑交互的
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        # decompress non-anchor
        depth_slice_nonanchor = decompress_nonanchor(self.depth_gaussian_conditional, depth_scales_nonanchor, depth_means_nonanchor, depth_decoder, depth_cdf, depth_cdf_lengths, depth_offsets)
        depth_y_hat_slice = depth_slice_nonanchor + depth_slice_anchor

        return rgb_y_hat_slice, depth_y_hat_slice

    def decompress_index1234(
        self,
        rgb_decoder,
        depth_decoder,
        rgb_y_hat_slices,
        depth_y_hat_slices,
        rgb_hyper_params,
        depth_hyper_params,
        idx,
        rgb_cdf,
        rgb_cdf_lengths,
        rgb_offsets,
        depth_cdf,
        depth_cdf_lengths,
        depth_offsets,
    ):
        # channel
        rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))  # 如果是这样作为上下文，有可能出现把原来的信息给挤掉了，因为channel数量没有改变
        depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))

        # rgb-anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](torch.cat([rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))  # 这里仍然可以添加另一个模态的信息
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        # decompress anchor
        rgb_slice_anchor = decompress_anchor(self.rgb_gaussian_conditional, rgb_scales_anchor, rgb_means_anchor, rgb_decoder, rgb_cdf, rgb_cdf_lengths, rgb_offsets)
        rgb_local_ctx = self.rgb_local_context[idx](rgb_slice_anchor)

        # Anchor
        depth_params_anchor = self.depth_entropy_parameters_anchor[idx](torch.cat([rgb_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        # decompress anchor
        depth_slice_anchor = decompress_anchor(self.depth_gaussian_conditional, depth_scales_anchor, depth_means_anchor, depth_decoder, depth_cdf, depth_cdf_lengths, depth_offsets)
        depth_local_ctx = self.depth_local_context[idx](depth_slice_anchor)

        # Non-anchor
        # local_ctx: [B,2 * C, H, W]
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[idx](torch.cat([rgb_local_ctx, depth_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1))

        # rgb_params_nonanchor = ckbd_merge(rgb_params_anchor, rgb_params_nonanchor)  # 为了弥补forward出现bpp的问题
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        rgb_slice_nonanchor = decompress_nonanchor(self.rgb_gaussian_conditional, rgb_scales_nonanchor, rgb_means_nonanchor, rgb_decoder, rgb_cdf, rgb_cdf_lengths, rgb_offsets)
        rgb_y_hat_slice = rgb_slice_nonanchor + rgb_slice_anchor
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)
        rgb_y_hat_slices.append(rgb_y_hat_slice)

        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx, depth_local_ctx, rgb_channel_ctx, depth_channel_ctx, rgb_hyper_params, depth_hyper_params], dim=1)
        )
        # depth_params_nonanchor = ckbd_merge(depth_params_anchor, depth_params_nonanchor)  # 为了弥补forward出现bpp的问题
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        depth_slice_nonanchor = decompress_nonanchor(self.depth_gaussian_conditional, depth_scales_nonanchor, depth_means_nonanchor, depth_decoder, depth_cdf, depth_cdf_lengths, depth_offsets)
        depth_y_hat_slice = depth_slice_nonanchor + depth_slice_anchor
        depth_y_hat_slices.append(depth_y_hat_slice)

        return rgb_y_hat_slices, depth_y_hat_slices

    def bin2img_united(self, rgb_y_strings, rgb_hyper_params, depth_y_strings, depth_hyper_params):
        rgb_y_hat_slices = []
        rgb_cdf = self.rgb_gaussian_conditional.quantized_cdf.tolist()
        rgb_cdf_lengths = self.rgb_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        rgb_offsets = self.rgb_gaussian_conditional.offset.reshape(-1).int().tolist()
        rgb_decoder = RansDecoder()
        rgb_decoder.set_stream(rgb_y_strings)

        depth_y_hat_slices = []
        depth_cdf = self.depth_gaussian_conditional.quantized_cdf.tolist()
        depth_cdf_lengths = self.depth_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        depth_offsets = self.depth_gaussian_conditional.offset.reshape(-1).int().tolist()
        depth_decoder = RansDecoder()
        depth_decoder.set_stream(depth_y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                rgb_y_hat_slice, depth_y_hat_slice = self.decompress_index0(
                    rgb_decoder, rgb_hyper_params, rgb_cdf, rgb_cdf_lengths, rgb_offsets, depth_decoder, depth_hyper_params, depth_cdf, depth_cdf_lengths, depth_offsets
                )
                rgb_y_hat_slices.append(rgb_y_hat_slice)

                depth_y_hat_slices.append(depth_y_hat_slice)

            else:
                rgb_y_hat_slices, depth_y_hat_slices = self.decompress_index1234(
                    rgb_decoder,
                    depth_decoder,
                    rgb_y_hat_slices,
                    depth_y_hat_slices,
                    rgb_hyper_params,
                    depth_hyper_params,
                    idx,
                    rgb_cdf,
                    rgb_cdf_lengths,
                    rgb_offsets,
                    depth_cdf,
                    depth_cdf_lengths,
                    depth_offsets,
                )

        rgb_y_hat = torch.cat(rgb_y_hat_slices, dim=1)
        depth_y_hat = torch.cat(depth_y_hat_slices, dim=1)

        return rgb_y_hat, depth_y_hat

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        rgb_updated = self.rgb_gaussian_conditional.update_scale_table(scale_table, force=force)
        depth_updated = self.depth_gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = rgb_updated & depth_updated | super().update(force=force)  # 更新entropybottleneck
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(self.rgb_gaussian_conditional, "rgb_gaussian_conditional", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        update_registered_buffers(self.depth_gaussian_conditional, "depth_gaussian_conditional", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        update_registered_buffers(self.rgb_entropy_bottleneck, "rgb_entropy_bottleneck", ["_quantized_cdf", "_offset", "_cdf_length"], state_dict)
        update_registered_buffers(self.depth_entropy_bottleneck, "depth_entropy_bottleneck", ["_quantized_cdf", "_offset", "_cdf_length"], state_dict)
        super().load_state_dict(state_dict)

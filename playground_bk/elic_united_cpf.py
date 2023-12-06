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


class ELIC_united_cpf(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch  # [8, 8, 8, 8, 16, 16, 32, 32, 96, 96]
        self.quant = config.quant  # noise or ste
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEXcro(N, M, act=nn.ReLU)
        # self.g_a = AnalysisTransformEXcross(N, M, act=nn.ReLU)
        self.g_s = SynthesisTransformEXcross(N, M, act=nn.ReLU)
        # Hyper Transform
        self.h_a = HyperAnalysisEXcro(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEXcro(N, M, act=nn.ReLU)
        # Channel Fusion Model
        self.rgb_local_context = nn.ModuleList(nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2) for i in range(len(slice_ch)))
        self.depth_local_context = nn.ModuleList(nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2) for i in range(len(slice_ch)))
        self.rgb_channel_context = nn.ModuleList(ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None for i in range(slice_num))
        self.depth_channel_context = nn.ModuleList(ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None for i in range(slice_num))

        # Use channel_ctx and hyper_params
        self.rgb_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        self.depth_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.rgb_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        self.depth_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
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

    def entropy_estimate(self, y, hyper_params, gaussian_conditional, local_context, channel_context, entropy_parameters_anchor, entropy_parameters_nonanchor):
        y_slices = [y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == "ste":
                    slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = gaussian_conditional.quantize(slice_anchor, "noise" if self.training else "dequantize")
                    slice_anchor = ckbd_anchor(slice_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = local_context[idx](slice_anchor)
                params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == "ste":
                    slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = gaussian_conditional.quantize(slice_nonanchor, "noise" if self.training else "dequantize")
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                channel_ctx = channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)  # 预测的时候是生成没有mask的，也就是nonanchor部分随便生成，然后再mask
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == "ste":
                    slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = gaussian_conditional.quantize(slice_anchor, "noise" if self.training else "dequantize")
                    slice_anchor = ckbd_anchor(slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = local_context[idx](slice_anchor)
                params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == "ste":
                    slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = gaussian_conditional.quantize(slice_nonanchor, "noise" if self.training else "dequantize")
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        return y_hat, y_likelihoods

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
        rgb_y_hat, rgb_y_likelihoods = self.entropy_estimate(
            rgb_y, rgb_hyper_params, self.rgb_gaussian_conditional, self.rgb_local_context, self.rgb_channel_context, self.rgb_entropy_parameters_anchor, self.rgb_entropy_parameters_nonanchor
        )
        depth_y_hat, depth_y_likelihoods = self.entropy_estimate(
            depth_y,
            depth_hyper_params,
            self.depth_gaussian_conditional,
            self.depth_local_context,
            self.depth_channel_context,
            self.depth_entropy_parameters_anchor,
            self.depth_entropy_parameters_nonanchor,
        )

        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        return {"x_hat": {"r": rgb_hat, "d": depth_hat}, "r_likelihoods": {"y": rgb_y_likelihoods, "z": rgb_z_likelihoods}, "d_likelihoods": {"y": depth_y_likelihoods, "z": depth_z_likelihoods}}

    def feature2bin(self, y, hyper_params, gaussian_conditional, local_context, channel_context, entropy_parameters_anchor, entropy_parameters_nonanchor):
        y_slices = [y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []

        cdf = gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)

                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = local_context[idx](slice_anchor)
                params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_slice_hat = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_slice_hat)

            else:
                # Anchor
                channel_ctx = channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = local_context[idx](slice_anchor)
                params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)
        return y_strings

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
        rgb_y_strings = self.feature2bin(
            rgb_y, rgb_hyper_params, self.rgb_gaussian_conditional, self.rgb_local_context, self.rgb_channel_context, self.rgb_entropy_parameters_anchor, self.rgb_entropy_parameters_nonanchor
        )
        depth_y_strings = self.feature2bin(
            depth_y,
            depth_hyper_params,
            self.depth_gaussian_conditional,
            self.depth_local_context,
            self.depth_channel_context,
            self.depth_entropy_parameters_anchor,
            self.depth_entropy_parameters_nonanchor,
        )

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

        rgb_y_hat = self.bin2img(
            rgb_y_strings, rgb_hyper_params, self.rgb_gaussian_conditional, self.rgb_local_context, self.rgb_channel_context, self.rgb_entropy_parameters_anchor, self.rgb_entropy_parameters_nonanchor
        )

        depth_y_hat = self.bin2img(
            depth_y_strings,
            depth_hyper_params,
            self.depth_gaussian_conditional,
            self.depth_local_context,
            self.depth_channel_context,
            self.depth_entropy_parameters_anchor,
            self.depth_entropy_parameters_nonanchor,
        )

        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time
        return {"x_hat": {"r": rgb_hat.clamp_(0, 1), "d": depth_hat.clamp_(0, 1)}, "cost_time": cost_time}

    def bin2img(self, y_strings, hyper_params, gaussian_conditional, local_context, channel_context, entropy_parameters_anchor, entropy_parameters_nonanchor):
        y_hat_slices = []
        cdf = gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = local_context[idx](slice_anchor)
                params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

            else:
                # Anchor
                channel_ctx = channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = local_context[idx](slice_anchor)
                params_nonanchor = entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        # torch.backends.cudnn.deterministic = False # 问题所在

        return y_hat

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

import os
import time

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import compute_metrics
from utils.utils import *


def compress_one_image(model, x, stream_path, H, W, img_name):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x)
    torch.cuda.synchronize()
    end_time = time.time()
    shape = out["shape"]
    os.makedirs(stream_path, exist_ok=True)
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    enc_time = end_time - start_time
    return bpp, enc_time


def compress_one_image_united(model, x, stream_path, H, W, img_name):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x[0], x[1])
    torch.cuda.synchronize()
    end_time = time.time()
    shape = out["shape"]
    os.makedirs(stream_path[0], exist_ok=True)
    os.makedirs(stream_path[1], exist_ok=True)

    rgb_output = os.path.join(stream_path[0], img_name)
    # print("rstring")
    with Path(rgb_output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["r_strings"])
    size = filesize(rgb_output)
    rgb_bpp = float(size) * 8 / (H * W)

    # print("dstring")
    depth_output = os.path.join(stream_path[1], img_name)
    with Path(depth_output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["d_strings"])
    size = filesize(depth_output)
    depth_bpp = float(size) * 8 / (H * W)

    enc_time = end_time - start_time
    return rgb_bpp, depth_bpp, enc_time


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(strings, shape)
    torch.cuda.synchronize()
    end_time = time.time()
    dec_time = end_time - start_time
    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    return x_hat, dec_time


def decompress_one_image_united(model, stream_path, img_name):
    rgb_output = os.path.join(stream_path[0], img_name)
    with Path(rgb_output).open("rb") as f:
        original_size = read_uints(f, 2)
        rgb_strings, shape = read_body(f)

    depth_output = os.path.join(stream_path[1], img_name)
    with Path(depth_output).open("rb") as f:
        original_size = read_uints(f, 2)
        depth_strings, shape = read_body(f)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(rgb_strings, depth_strings, shape)
    torch.cuda.synchronize()
    end_time = time.time()
    dec_time = end_time - start_time
    rgb_x_hat = out["x_hat"]["r"]
    rgb_x_hat = rgb_x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]

    depth_x_hat = out["x_hat"]["d"]
    depth_x_hat = depth_x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    return rgb_x_hat, depth_x_hat, dec_time


def test_model(test_dataloader, net, logger_test, save_dir, epoch):
    net.eval()
    device = next(net.parameters()).device

    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)

            from torchvision import transforms

            img = transforms.CenterCrop((448, 576))(img)
            img_pad = img

            B, C, H, W = img.shape
            # pad_h = 0
            # pad_w = 0
            # if H % 64 != 0:
            #     pad_h = 64 * (H // 64 + 1) - H
            # if W % 64 != 0:
            #     pad_w = 64 * (W // 64 + 1) - W

            # img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
            # net.update_resolutions(img_pad.size(2) // 16, img_pad.size(3) // 16)
            bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=os.path.join(save_dir, "bin"), H=H, W=W, img_name=str(i))
            x_hat, dec_time = decompress_one_image(model=net, stream_path=os.path.join(save_dir, "bin"), img_name=str(i))
            p, m = compute_metrics(x_hat.clamp_(0, 1), img.clamp_(0, 1))
            if x_hat.shape[1] == 1:
                depth = x_hat.cpu().squeeze().numpy().astype("uint16")
                cv2.imwrite(os.path.join(save_dir, "%03d_rec_16bit.png" % i), depth)

            rec = torch2img(x_hat)
            img = torch2img(img)
            # img.save(os.path.join(save_dir, '%03d_gt.png' % i))
            rec.save(os.path.join(save_dir, "%03d_rec.png" % i))
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            avg_bpp.update(bpp)
            avg_deocde_time.update(dec_time)
            avg_encode_time.update(enc_time)
            logger_test.info(f"Image[{i}] | " f"Bpp loss: {bpp:.4f} | " f"PSNR: {p:.4f} | " f"MS-SSIM: {m:.4f} | " f"Encoding Latency: {enc_time:.4f} | " f"Decoding latency: {dec_time:.4f}")
    logger_test.info(
        f"Epoch:[{epoch}] | "
        f"Avg Bpp: {avg_bpp.avg:.7f} | "
        f"Avg PSNR: {avg_psnr.avg:.7f} | "
        f"Avg MS-SSIM: {avg_ms_ssim.avg:.7f} | "
        f"Avg Encoding Latency: {avg_encode_time.avg:.6f} | "
        f"Avg Decoding latency: {avg_deocde_time.avg:.6f}"
    )


def test_model_united(test_dataloader, net, logger_test, save_dir, epoch):
    net.eval()
    device = next(net.parameters()).device

    avg_rgb_psnr = AverageMeter()
    avg_rgb_ms_ssim = AverageMeter()
    avg_rgb_bpp = AverageMeter()
    avg_depth_psnr = AverageMeter()
    avg_depth_ms_ssim = AverageMeter()
    avg_depth_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()

    with torch.no_grad():
        for i, (rgb, depth) in enumerate(test_dataloader):
            rgb = rgb.to(device)
            depth = depth.to(device)
            from torchvision import transforms

            rgb = transforms.CenterCrop((448, 576))(rgb)
            depth = transforms.CenterCrop((448, 576))(depth)  # 因为白边是255，但是padding是0，不一样。但是实际使用的时候，还是需要考虑padding，还是考虑一下怎么解决才是正道。
            rgb_pad = rgb
            depth_pad = depth

            # 为什么要进行padding？
            B, C, H, W = rgb.shape
            # pad_h = 0
            # pad_w = 0
            # if H % 64 != 0:
            #     pad_h = 64 * (H // 64 + 1) - H
            # if W % 64 != 0:
            #     pad_w = 64 * (W // 64 + 1) - W

            # rgb_pad = F.pad(rgb, (0, pad_w, 0, pad_h), mode="constant", value=0)

            # depth_pad = F.pad(depth, (0, pad_w, 0, pad_h), mode="constant", value=0)
            # rgb_pad = rgb
            # depth_pad = depth

            rgb_bpp, depth_bpp, enc_time = compress_one_image_united(
                model=net, x=(rgb_pad, depth_pad), stream_path=(os.path.join(save_dir, "rgb_bin"), os.path.join(save_dir, "depth_bin")), H=H, W=W, img_name=str(i)
            )
            rgb_x_hat, depth_x_hat, dec_time = decompress_one_image_united(model=net, stream_path=(os.path.join(save_dir, "rgb_bin"), os.path.join(save_dir, "depth_bin")), img_name=str(i))

            rgb_p, rgb_m = compute_metrics(rgb_x_hat.clamp_(0, 1), rgb.clamp_(0, 1))
            depth_p, depth_m = compute_metrics(depth_x_hat.clamp_(0, 1), depth.clamp_(0, 1))

            avg_rgb_psnr.update(rgb_p)
            avg_rgb_ms_ssim.update(rgb_m)
            avg_rgb_bpp.update(rgb_bpp)
            rec = torch2img(rgb_x_hat)
            if not os.path.exists(os.path.join(save_dir, "rgb_rec")):
                os.makedirs(os.path.join(save_dir, "rgb_rec"), exist_ok=True)
            rec.save(os.path.join(save_dir, "rgb_rec", "%03d_rec.png" % i))

            avg_depth_psnr.update(depth_p)
            avg_depth_ms_ssim.update(depth_m)
            avg_depth_bpp.update(depth_bpp)
            rec = torch2img(depth_x_hat)
            if not os.path.exists(os.path.join(save_dir, "depth_rec")):
                os.makedirs(os.path.join(save_dir, "depth_rec"), exist_ok=True)
            rec.save(os.path.join(save_dir, "depth_rec", "%03d_rec.png" % i))
            depth = depth_x_hat.cpu().squeeze().numpy().astype("uint16")
            cv2.imwrite(os.path.join(save_dir, "depth_rec", "%03d_rec_16bit.png" % i), depth)

            avg_deocde_time.update(dec_time)
            avg_encode_time.update(enc_time)
            logger_test.info(
                f"Image[{i}] | "
                f"rBpp loss: {rgb_bpp:.4f} | "
                f"dBpp loss: {depth_bpp:.4f} | "
                f"rPSNR: {rgb_p:.4f} | "
                f"dPSNR: {depth_p:.4f} | "
                f"rMS-SSIM: {rgb_m:.4f} | "
                f"sMS-SSIM: {depth_m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding latency: {dec_time:.4f}"
            )
    logger_test.info(
        f"Epoch:[{epoch}] | "
        f"Avg rBpp: {avg_rgb_bpp.avg:.7f} | "
        f"Avg dBpp: {avg_depth_bpp.avg:.7f} | "
        f"Avg rPSNR: {avg_rgb_psnr.avg:.7f} | "
        f"Avg dPSNR: {avg_depth_psnr.avg:.7f} | "
        f"Avg rMS-SSIM: {avg_rgb_ms_ssim.avg:.7f} | "
        f"Avg dMS-SSIM: {avg_depth_ms_ssim.avg:.7f} | "
        f"Avg Encoding Latency: {avg_encode_time.avg:.6f} | "
        f"Avg Decoding latency: {avg_deocde_time.avg:.6f}"
    )

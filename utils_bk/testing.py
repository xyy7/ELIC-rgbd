import os
import time

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import compute_metrics,AverageMeter
from utils.IOutils import *


def window_partition(x, window_size=4):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = windows.permute(0, 3, 1, 2)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    windows = windows.permute(0, 2, 3, 1)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = x.permute(0, 3, 1, 2)
    return x


def compress_one_image(model, x, stream_path, H, W, img_name, pad_H=None, pad_W=None):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x)
    torch.cuda.synchronize()
    end_time = time.time()
    shape = out["shape"]
    os.makedirs(stream_path, exist_ok=True)
    output = os.path.join(stream_path, img_name)
    if pad_H is None:
        pad_H=H
        pad_W=W
    with Path(output).open("wb") as f:
        write_uints(f, (pad_H, pad_W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    enc_time = end_time - start_time

    return bpp, enc_time


def compress_master(model, x, aux, aux_out, stream_path, padH, padW, H, W, img_name):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x, aux, aux_out)
    torch.cuda.synchronize()
    end_time = time.time()
    shape = out["shape"]
    os.makedirs(stream_path, exist_ok=True)
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (padH, padW))
        write_body(f, shape, out["strings"])

    size = filesize(output) + 128  # 64个4字节float的beta+64个4字节float的gamma
    bpp = float(size) * 8 / (H * W * len(x) ** 2)
    enc_time = end_time - start_time
    return bpp, enc_time, out["beta"], out["gamma"]


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


def decompress_one_image(model, stream_path, img_name, mode="reflect0", return_mid=False):
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
    if mode.find("0") != -1:
        x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    else:
        x_hat = crop(x_hat, original_size)
    if return_mid:
        return x_hat, dec_time, out  # out["x_hat"],out["up1"],out["up2"],out["up3"]
    return x_hat, dec_time


def decompress_master(model, aux, aux_out, beta, gamma, stream_path, img_name, mode="reflect0"):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(strings, shape, beta, gamma, aux, aux_out)
    torch.cuda.synchronize()
    end_time = time.time()
    dec_time = end_time - start_time
    x_hat = out["x_hat"]
    if mode.find("0") != -1:
        x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    else:
        x_hat = crop(x_hat, original_size)
    return x_hat, dec_time


def decompress_one_image_united(model, stream_path, img_name, mode="reflect0"):
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
    depth_x_hat = out["x_hat"]["d"]
    if mode.find("0") != -1:
        rgb_x_hat = rgb_x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
        depth_x_hat = depth_x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    else:
        rgb_x_hat = crop(rgb_x_hat, original_size)
        depth_x_hat = crop(depth_x_hat, original_size)
    return rgb_x_hat, depth_x_hat, dec_time


def pad(x, p=2**6, mode="reflect"):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        # mode="constant",
        mode=mode,
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom), mode="constant", value=0)


def test_model(test_dataloader, net, logger_test, save_dir, epoch, mode="reflect"):
    net.eval()
    device = next(net.parameters()).device
    total_params = sum(p.numel() for p in net.parameters())
    print("params:", total_params)
    time.sleep(10)

    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()

    padding = True
    # mode = "reflect"
    if not padding:
        save_dir = save_dir + "-ct"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = save_dir + "-padding-" + mode
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth_rec"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "rgb_rec"), exist_ok=True)

    with torch.no_grad():
        for i, (img, img_name) in enumerate(test_dataloader):
            img = img.to(device)
            if not padding:
                from torchvision import transforms

                img = transforms.CenterCrop((448, 576))(img)
                img_pad = img
            else:
                B, C, H, W = img.shape
                if mode.find("0") != -1:
                    pad_h = 0
                    pad_w = 0
                    if H % 64 != 0:
                        pad_h = 64 * (H // 64 + 1) - H
                    if W % 64 != 0:
                        pad_w = 64 * (W // 64 + 1) - W
                    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode=mode[:-1], value=0)  # 全都是旁边的padding
                else:
                    img_pad = pad(img, mode=mode[:-1])

            B, C, H, W = img.shape
            if C == 1:
                bpp, enc_time = compress_one_image(
                    model=net, x=img_pad, stream_path=os.path.join(save_dir, "depth_bin"), H=H, W=W, img_name=img_name[0]
                )  # 这样在进行压缩率实验的时候,没什么问题,但是在进行miou实验的时候,就有麻烦(对应不上)
                x_hat, dec_time = decompress_one_image(model=net, stream_path=os.path.join(save_dir, "depth_bin"), img_name=img_name[0], mode=mode)  # 可能会转化成tuple
            else:
                bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=os.path.join(save_dir, "rgb_bin"), H=H, W=W, img_name=img_name[0])  # 这样在进行压缩率实验的时候,没什么问题,但是在进行miou实验的时候,就有麻烦(对应不上)
                x_hat, dec_time = decompress_one_image(model=net, stream_path=os.path.join(save_dir, "rgb_bin"), img_name=img_name[0], mode=mode)  # 可能会转化成tuple

            p, m = compute_metrics(x_hat.clamp_(0, 1), img.clamp_(0, 1))
            if x_hat.shape[1] == 1:
                if save_dir.find("sun") != -1:
                    depth = x_hat * 100000
                else:
                    depth = x_hat * 10000
                depth = depth.cpu().squeeze().numpy().astype("uint16")
                cv2.imwrite(os.path.join(save_dir, "depth_rec", f"{img_name[0]}_rec_16bit.png"), depth)

            rec = torch2img(x_hat)
            img = torch2img(img)
            if x_hat.shape[1] == 1:
                rec.save(os.path.join(save_dir, "depth_rec", f"{img_name[0]}_rec.png"))
            else:
                rec.save(os.path.join(save_dir, "rgb_rec", f"{img_name[0]}_rec.png"))
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

    def writeInfo(dir, file):
        files = os.listdir(dir)
        files.sort()
        with open(file, "w") as f:
            for fn in files:
                f.write(f"{dir}/{fn}\n")

    writeInfo(os.path.join(save_dir, "depth_rec"), os.path.join(save_dir, "test_depth.txt"))
    writeInfo(os.path.join(save_dir, "rgb_rec"), os.path.join(save_dir, "test_rgb.txt"))


def test_master(test_dataloader, net, net_single, logger_test, save_dir, epoch, mode="reflect"):
    net.eval()
    net_single.eval()
    device = next(net.parameters()).device
    total_params = sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in net_single.parameters())
    print("params:", total_params)
    time.sleep(10)

    avg_rgb_psnr = AverageMeter()
    avg_rgb_ms_ssim = AverageMeter()
    avg_rgb_bpp = AverageMeter()
    avg_depth_psnr = AverageMeter()
    avg_depth_ms_ssim = AverageMeter()
    avg_depth_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()

    padding = True
    # mode = "reflect"
    if not padding:
        save_dir = save_dir + "-ct"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = save_dir + "-padding-" + mode
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth_rec"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "rgb_rec"), exist_ok=True)

    with torch.no_grad():
        for i, (rgb, depth, rgb_imgname, depth_imgname) in enumerate(test_dataloader):
            rgb = rgb.to(device)
            depth = depth.to(device)
            img_name = rgb_imgname
            # 判断哪一个是aux，那个是master
            if save_dir.find("depth") != -1:
                img = depth
                aux = rgb
            else:
                img = rgb
                aux = depth

            pad_num = 256
            if not padding:
                from torchvision import transforms

                img = transforms.CenterCrop((448, 576))(img)
                img_pad = img

                aux = transforms.CenterCrop((448, 576))(aux)
                aux_pad = aux
            else:
                B, C, H, W = img.shape
                if mode.find("0") != -1:
                    pad_h = 0
                    pad_w = 0
                    if H % pad_num != 0:
                        pad_h = pad_num * (H // pad_num + 1) - H
                    if W % pad_num != 0:
                        pad_w = pad_num * (W // pad_num + 1) - W
                    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode=mode[:-1], value=0)  # 全都是旁边的padding
                    aux_pad = F.pad(aux, (0, pad_w, 0, pad_h), mode=mode[:-1], value=0)  # 全都是旁边的padding
                else:
                    img_pad = pad(img, p=pad_num, mode=mode[:-1])
                    aux_pad = pad(aux, p=pad_num, mode=mode[:-1])

            B, C, H, W = img.shape
            _, C_aux, H, W = aux.shape
            _, _, H_pad, W_pad = aux_pad.shape
            # 分块与恢复
            print(img.shape, aux.shape, img_pad.shape, aux_pad.shape)
            # img_pad = img_pad.reshape(-1, C, pad_num, pad_num)
            # aux_pad = aux_pad.reshape(-1, C_aux, pad_num, pad_num)
            img_pad = window_partition(img_pad, pad_num)
            aux_pad = window_partition(aux_pad, pad_num)
            print(img.shape, aux.shape, img_pad.shape, aux_pad.shape)

            if C == 1:
                aux_bin = "rgb_bin"
                master_bin = "depth_bin"
            else:
                aux_bin = "depth_bin"
                master_bin = "rgb_bin"

            aux_bpp, aux_enc_time = compress_one_image(
                model=net_single, x=aux_pad, stream_path=os.path.join(save_dir, aux_bin), pad_H=pad_num, pad_W=pad_num, H=H, W=W, img_name=img_name[0]
            )  # 这样在进行压缩率实验的时候,没什么问题,但是在进行miou实验的时候,就有麻烦(对应不上)
            aux_hat, aux_dec_time, aux_out = decompress_one_image(model=net_single, stream_path=os.path.join(save_dir, aux_bin), img_name=img_name[0], mode=mode, return_mid=True)  # 可能会转化成tuple

            bpp, enc_time, beta, gamma = compress_master(
                model=net, x=img_pad, aux=aux_hat, aux_out=aux_out, stream_path=os.path.join(save_dir, master_bin), padH=pad_num, padW=pad_num, H=H, W=W, img_name=img_name[0]
            )  # 这样在进行压缩率实验的时候,没什么问题,但是在进行miou实验的时候,就有麻烦(对应不上)
            x_hat, dec_time = decompress_master(
                model=net, aux=aux_hat, aux_out=aux_out, beta=beta, gamma=gamma, stream_path=os.path.join(save_dir, master_bin), img_name=img_name[0], mode=mode
            )  # 可能会转化成tuple

            # 分块与恢复
            # x_hat = x_hat.reshape(B, C, H_pad, W_pad)[:, :, 0:H, 0:W]
            # aux_hat = aux_hat.reshape(B, C_aux, H_pad, W_pad)[:, :, 0:H, 0:W]
            x_hat = window_reverse(x_hat, pad_num, H_pad, W_pad)[:, :, 0:H, 0:W]
            aux_hat = window_reverse(aux_hat, pad_num, H_pad, W_pad)[:, :, 0:H, 0:W]

            p, m = compute_metrics(x_hat.clamp_(0, 1), img.clamp_(0, 1))
            p_aux, m_aux = compute_metrics(aux_hat.clamp_(0, 1), aux.clamp_(0, 1))

            if x_hat.shape[1] == 1:
                if save_dir.find("sun") != -1:
                    depth = x_hat * 100000
                else:
                    depth = x_hat * 10000
                depth = depth.cpu().squeeze().numpy().astype("uint16")
                cv2.imwrite(os.path.join(save_dir, "depth_rec", f"{img_name[0]}_rec_16bit.png"), depth)

            rec = torch2img(x_hat)
            aux = torch2img(aux_hat)
            if x_hat.shape[1] == 1:
                rec.save(os.path.join(save_dir, "depth_rec", f"{img_name[0]}_rec.png"))
                aux.save(os.path.join(save_dir, "rgb_rec", f"{img_name[0]}_rec.png"))
            else:
                rec.save(os.path.join(save_dir, "rgb_rec", f"{img_name[0]}_rec.png"))
                aux.save(os.path.join(save_dir, "depth_rec", f"{img_name[0]}_rec.png"))

            if C == 1:
                rgb_bpp = aux_bpp
                depth_bpp = bpp
                rgb_p = p_aux
                depth_p = p
                rgb_m = m_aux
                depth_m = m
            else:
                rgb_bpp = bpp
                depth_bpp = aux_bpp
                rgb_p = p
                depth_p = p_aux
                rgb_m = m
                depth_m = m_aux

            avg_depth_psnr.update(depth_p)
            avg_depth_ms_ssim.update(depth_m)
            avg_depth_bpp.update(depth_bpp)
            avg_rgb_psnr.update(rgb_p)
            avg_rgb_ms_ssim.update(rgb_m)
            avg_rgb_bpp.update(rgb_bpp)
            avg_deocde_time.update(dec_time + aux_dec_time)  # 计算的就是总时间
            avg_encode_time.update(enc_time + aux_enc_time)

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

    def writeInfo(dir, file):
        files = os.listdir(dir)
        files.sort()
        with open(file, "w") as f:
            for fn in files:
                f.write(f"{dir}/{fn}\n")

    writeInfo(os.path.join(save_dir, "depth_rec"), os.path.join(save_dir, "test_depth.txt"))
    writeInfo(os.path.join(save_dir, "rgb_rec"), os.path.join(save_dir, "test_rgb.txt"))


def test_model_united(test_dataloader, net, logger_test, save_dir, epoch, mode="reflect"):
    net.eval()
    device = next(net.parameters()).device
    total_params = sum(p.numel() for p in net.parameters())
    print("params:", total_params)
    time.sleep(10)

    avg_rgb_psnr = AverageMeter()
    avg_rgb_ms_ssim = AverageMeter()
    avg_rgb_bpp = AverageMeter()
    avg_depth_psnr = AverageMeter()
    avg_depth_ms_ssim = AverageMeter()
    avg_depth_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()

    padding = True
    if not padding:
        save_dir = save_dir + "-ct"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = save_dir + "-padding-" + mode
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (rgb, depth, rgb_imgname, depth_imgname) in enumerate(test_dataloader):
            rgb = rgb.to(device)
            depth = depth.to(device)

            if not padding:
                from torchvision import transforms

                rgb = transforms.CenterCrop((448, 576))(rgb)  # 为什么是padding 2**6,因为下采样6次,否则就需要在下采样的时候,写两份代码.
                depth = transforms.CenterCrop((448, 576))(depth)  # 因为白边是255，但是padding是0，不一样。但是实际使用的时候，还是需要考虑padding，还是考虑一下怎么解决才是正道。
                rgb_pad = rgb
                depth_pad = depth

            else:
                B, C, H, W = rgb.shape
                # print("padding mode", mode)
                if mode.find("0") != -1:  # 0代表向右或者向左
                    real_mode = mode[:-1]
                    pad_h = 0
                    pad_w = 0
                    if H % 64 != 0:
                        pad_h = 64 * (H // 64 + 1) - H
                    if W % 64 != 0:
                        pad_w = 64 * (W // 64 + 1) - W

                    rgb_pad = F.pad(rgb, (0, pad_w, 0, pad_h), mode=real_mode, value=0)
                    depth_pad = F.pad(depth, (0, pad_w, 0, pad_h), mode=real_mode, value=0)
                else:
                    real_mode = mode[:-1]
                    rgb_pad = pad(rgb, mode=real_mode)
                    depth_pad = pad(depth, mode=real_mode)
            B, C, H, W = rgb.shape

            rgb_bpp, depth_bpp, enc_time = compress_one_image_united(
                model=net, x=(rgb_pad, depth_pad), stream_path=(os.path.join(save_dir, "rgb_bin"), os.path.join(save_dir, "depth_bin")), H=H, W=W, img_name=rgb_imgname[0]
            )
            rgb_x_hat, depth_x_hat, dec_time = decompress_one_image_united(
                model=net, stream_path=(os.path.join(save_dir, "rgb_bin"), os.path.join(save_dir, "depth_bin")), img_name=rgb_imgname[0], mode=mode
            )

            rgb_p, rgb_m = compute_metrics(rgb_x_hat.clamp_(0, 1), rgb.clamp_(0, 1))
            depth_p, depth_m = compute_metrics(depth_x_hat.clamp_(0, 1), depth.clamp_(0, 1))

            avg_rgb_psnr.update(rgb_p)
            avg_rgb_ms_ssim.update(rgb_m)
            avg_rgb_bpp.update(rgb_bpp)
            rec = torch2img(rgb_x_hat)
            if not os.path.exists(os.path.join(save_dir, "rgb_rec")):
                os.makedirs(os.path.join(save_dir, "rgb_rec"), exist_ok=True)
            # rgb重建
            rec.save(os.path.join(save_dir, "rgb_rec", f"{rgb_imgname[0]}_rec.png"))

            avg_depth_psnr.update(depth_p)
            avg_depth_ms_ssim.update(depth_m)
            avg_depth_bpp.update(depth_bpp)
            rec = torch2img(depth_x_hat)
            if not os.path.exists(os.path.join(save_dir, "depth_rec")):
                os.makedirs(os.path.join(save_dir, "depth_rec"), exist_ok=True)
            # depth重建
            rec.save(os.path.join(save_dir, "depth_rec", f"{rgb_imgname[0]}_rec.png"))
            # 写入16bit
            if save_dir.find("sun") != -1:
                depth = depth_x_hat * 100000
            else:
                depth = depth_x_hat * 10000
            depth = depth.cpu().squeeze().numpy().astype("uint16")
            cv2.imwrite(os.path.join(save_dir, "depth_rec", f"{rgb_imgname[0]}_rec_16bit.png"), depth)

            # torch2img(rgb_pad).save("rgbpad.png")
            # torch2img(depth_pad).save("depthpad.png")
            # print(os.path.curdir + "\\" + "padding-img")
            # exit()

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
    with open("padding_mode.txt", "a") as file:
        file.write(
            f"{save_dir} Epoch:[{epoch}] | "
            f"Avg rBpp: {avg_rgb_bpp.avg:.7f} | "
            f"Avg dBpp: {avg_depth_bpp.avg:.7f} | "
            f"Avg rPSNR: {avg_rgb_psnr.avg:.7f} | "
            f"Avg dPSNR: {avg_depth_psnr.avg:.7f} | "
            f"Avg rMS-SSIM: {avg_rgb_ms_ssim.avg:.7f} | "
            f"Avg dMS-SSIM: {avg_depth_ms_ssim.avg:.7f} | "
            f"Avg Encoding Latency: {avg_encode_time.avg:.6f} | "
            f"Avg Decoding latency: {avg_deocde_time.avg:.6f}\n"
        )

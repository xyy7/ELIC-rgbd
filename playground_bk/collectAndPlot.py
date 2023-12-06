import os
import re


# 获取相对应的模态的实验结果文件夹
# mode可以是模态名称，也可以是模型名称
def get_dir_list(root, mode):
    dirs = os.listdir(root)
    # print(dirs)
    dirs = [d for d in dirs if d.find(mode) != -1]
    dirs.sort()
    return dirs


# 根据log最后一行，获取相对应的数值
# 23-09-12 14:30:41.979 - INFO: Epoch:[580] | Avg Bpp: 0.0134539 | Avg PSNR: 45.4748826 | Avg MS-SSIM: 0.9937113 | Avg Encoding Latency: 0.116185 | Avg Decoding latency: 0.168879
def get_one_metrics(filename):
    with open(filename, encoding="utf8") as file:
        contents = file.readlines()[-1]
        pattern = r"INFO: Epoch:\[\d*?\] \| Avg Bpp: (\d+\.\d*?) \| Avg PSNR: (\d+\.\d*?) \| Avg MS-SSIM: (\d+\.\d*?) \| Avg Encoding Latency: (\d+\.\d*?) \| Avg Decoding latency: (\d+\.\d*)"
        res = re.findall(pattern, contents)
        res = [float(r.strip()) for r in res[0]]
    # print(res)
    return res


def get_all_metrics(root, dirs):
    bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list = [], [], [], [], []
    for dir in dirs:
        exp_dirs = os.listdir(os.path.join(root, dir))
        exp_dirs = [d for d in exp_dirs if d.find("test_epoch") != -1]  # 只取最新的log
        exp_dirs.sort()
        # print(exp_dirs)

        # 防止有的没有eval
        if len(exp_dirs) == 0:
            break

        bpp, psnr, ssim, enc_time, dec_time = [0] * 5
        # 定义时间提取的正则表达式模式
        pattern = r"(\d{6}-\d{6})"

        # 提取时间并排序
        exp_dirs = sorted(exp_dirs, key=lambda x: re.search(pattern, x).group(1), reverse=True)
        last_psnr = 0
        last_ssim = 0

        for expd in exp_dirs:
            try:
                bpp, psnr, ssim, enc_time, dec_time = get_one_metrics(os.path.join(root, dir, expd))
                print(expd)
                if last_psnr != psnr or last_ssim != ssim:
                    last_psnr = psnr
                    last_ssim = ssim
                else:
                    if os.path.exists(os.path.join(root, dir, expd)):
                        os.remove(os.path.join(root, dir, expd))
                break
            except:
                print(expd, "not have been tested")
                if os.path.exists(os.path.join(root, dir, expd)):
                    os.remove(os.path.join(root, dir, expd))
                continue
        psnr_list.append(psnr)
        bpp_list.append(bpp)
        ssim_list.append(ssim)
        enc_time_list.append(enc_time)
        dec_time_list.append(dec_time)
        # exit()

    print_result((dirs, bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list))
    return [dirs, bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list]


## 计算总bpp下metric
# def get_total_metrics(rgb_metrics,
#                       depth_metrics,
#                       rgb_offset=0,
#                       depth_offset=0,
#                       modelname='ELIC'):
#     rgb_metrics = [m[rgb_offset:rgb_offset + 4] for m in rgb_metrics]
#     depth_metrics = [m[depth_offset:depth_offset + 4] for m in depth_metrics]
#     bpp_total = [r + d for r, d in zip(rgb_metrics[1], depth_metrics[1])]
#     rgb_metrics[1] = bpp_total
#     depth_metrics[1] = bpp_total
#     print_result(rgb_metrics)
#     print_result(depth_metrics)
#     print_result_for_draw(rgb_metrics, modelname, 'r')
#     print_result_for_draw(depth_metrics, modelname, 'd')
#     return rgb_metrics, depth_metrics

# def print_result_for_draw(result, modelname, mode='r'):
#     metric_name = ['bpp', 'psnr', 'ssim', 'enc_t', 'dec_t']
#     metric_name = [mode + modelname + '_' + m for m in metric_name]
#     for i, metriclist in enumerate(result):
#         if i == 0: continue
#         outstr = metric_name[i - 1] + " = ["
#         for metric in metriclist:
#             outstr += f'{metric:.4}, '
#         outstr += ']'
#         print(outstr)
#     print()


def print_result(result):
    dir_list, bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list = result
    print("model\t\t\t bpp\t\t psnr\t\t ssim\t\t enc_time\t dec_time")
    for i, _ in enumerate(bpp_list):
        print(f"{dir_list[i]}\t {bpp_list[i]}\t {psnr_list[i]}\t {ssim_list[i]}\t {enc_time_list[i]}\t {dec_time_list[i]}")
    print()


def get_total_metrics(rgb_metrics, depth_metrics, rgb_offset=0, depth_offset=0, modelname="ELIC"):
    rgb_metrics = [m[rgb_offset : rgb_offset + 4] for m in rgb_metrics]
    depth_metrics = [m[depth_offset : depth_offset + 4] for m in depth_metrics]

    outstr = f'"{modelname}": ' + "{\n"
    # 打印各自
    outstr += print_result_for_draw(rgb_metrics, "r")
    outstr += print_result_for_draw(depth_metrics, "d")
    # 打印总共
    outstr += '"total_bpp"' + ": ["
    bpp_total = [r + d for r, d in zip(rgb_metrics[1], depth_metrics[1])]
    for total_bpp in bpp_total:
        outstr += f"{total_bpp:.4}, "
    outstr += "],\n"
    outstr += "},\n"
    print(outstr)

    rgb_metrics[1] = bpp_total
    depth_metrics[1] = bpp_total
    print_result(rgb_metrics)
    print_result(depth_metrics)

    return rgb_metrics, depth_metrics


def print_result_for_draw(result, mode="r"):
    metric_name = ["bpp", "psnr", "ssim", "enc_t", "dec_t"]
    metric_name = [mode + m for m in metric_name]
    outstr = ""
    for i, metriclist in enumerate(result):
        if i == 0:
            continue
        outstr += f'"{metric_name[i - 1]}"' + ": ["
        for metric in metriclist:
            outstr += f"{metric:.4}, "
        outstr += "],\n"
    return outstr


def plot_one():
    pass


if __name__ == "__main__":
    root = "/home/xyy/ELIC/experiments"
    # 主要是为了方便仅仅打印模型名称
    # rgb_dirs = get_dir_list(root, "rgb")
    # depth_dirs = get_dir_list(root, "depth")

    # rgb_dirs = get_dir_list(root, "nyuv2_rgb_ckbd")
    # depth_dirs = get_dir_list(root, "nyuv2_depth_ckbd")

    # rgb_dirs = get_dir_list(root, "sunrgbd_rgb_ELIC")
    # depth_dirs = get_dir_list(root, "sunrgbd_depth_ELIC")

    rgb_dirs = get_dir_list(root, "sunrgbd_rgb_ckbd")
    depth_dirs = get_dir_list(root, "sunrgbd_depth_ckbd")

    rgb_metrics = get_all_metrics(root, rgb_dirs)
    depth_metrics = get_all_metrics(root, depth_dirs)

    # get_total_metrics(rgb_metrics, depth_metrics, 0, 0, "ELIC")
    # get_total_metrics(rgb_metrics, depth_metrics, 0, 0, "ELIC(CVPR22)")
    get_total_metrics(rgb_metrics, depth_metrics, 0, 0, "ckbd(CVPR21)")

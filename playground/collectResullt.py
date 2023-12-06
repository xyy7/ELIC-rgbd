import os
import re


# mode可以是模态名称，也可以是模型名称
def collect_test_dirs(root, mode):
    dirs = os.listdir(root)
    dirs = [d for d in dirs if d.find(mode) != -1]
    dirs.sort()
    return dirs


# 根据log最后一行，获取相对应的数值
# 23-09-18 18:08:10.491 - INFO: Epoch:[232] | Avg rBpp: 0.1566178 | Avg dBpp: 0.0115115 | Avg rPSNR: 34.3340977 | Avg dPSNR: 44.4662399 | Avg rMS-SSIM: 0.9712246 | Avg dMS-SSIM: 0.9943522 | Avg Encoding Latency: 0.256585 | Avg Decoding latency: 0.446901
def get_metrics_from_one_united_file(filename):
    with open(filename, encoding="utf8") as file:
        contents = file.readlines()[-1]
        pattern = r"INFO: Epoch:\[\d*?\] \| Avg rBpp: (\d+\.\d*) \| Avg dBpp: (\d+\.\d*) \| Avg rPSNR: (\d+\.\d*) \| Avg dPSNR: (\d+\.\d*) \| Avg rMS-SSIM: (\d+\.\d*) \| Avg dMS-SSIM: (\d+\.\d*) \| Avg Encoding Latency: (\d+\.\d*) \| Avg Decoding latency: (\d+\.\d*)"
        res = re.findall(pattern, contents)
        res = [float(r.strip()) for r in res[0]]
    return res


# 根据log最后一行，获取相对应的数值
# 23-09-12 14:30:41.979 - INFO: Epoch:[580] | Avg Bpp: 0.0134539 | Avg PSNR: 45.4748826 | Avg MS-SSIM: 0.9937113 | Avg Encoding Latency: 0.116185 | Avg Decoding latency: 0.168879
def get_metrics_from_one_single_file(filename):
    with open(filename, encoding="utf8") as file:
        contents = file.readlines()[-1]
        pattern = r"INFO: Epoch:\[\d*?\] \| Avg Bpp: (\d+\.\d*?) \| Avg PSNR: (\d+\.\d*?) \| Avg MS-SSIM: (\d+\.\d*?) \| Avg Encoding Latency: (\d+\.\d*?) \| Avg Decoding latency: (\d+\.\d*)"
        res = re.findall(pattern, contents)
        res = [float(r.strip()) for r in res[0]]
    # print(res)
    return res


def get_metrics_from_collected_united_dirs(root, dirs):
    rbpp_list, dbpp_list, rpsnr_list, dpsnr_list, rssim_list, dssim_list, enc_time_list, dec_time_list = [
        [] for _ in range(8)
    ]
    dir_list = []
    for dir in dirs:
        exp_dir = os.listdir(os.path.join(root, dir))
        test_file_list = [d for d in exp_dir if d.find("test_epoch") != -1]
        test_file_list.sort()

        if len(test_file_list) == 0:
            print(test_file_list, "not have been tested")
            continue

        rbpp, dbpp, rpsnr, dpsnr, rssim, dssim, enc_time, dec_time = [0] * 8

        # 定义时间提取的正则表达式模式
        pattern = r"(\d{6}-\d{6})"
        test_file_list = sorted(test_file_list, key=lambda x: re.search(pattern, x).group(1), reverse=True)
        for file in test_file_list:
            try:
                rbpp, dbpp, rpsnr, dpsnr, rssim, dssim, enc_time, dec_time = get_metrics_from_one_united_file(
                    os.path.join(root, dir, file)
                )
                break
            except:
                print(file, "not have been tested")
                if os.path.exists(os.path.join(root, dir, file)):
                    os.remove(os.path.join(root, dir, file))
                continue
        dir_list.append(dir)
        dpsnr_list.append(dpsnr)
        rpsnr_list.append(rpsnr)
        dbpp_list.append(dbpp)
        rbpp_list.append(rbpp)
        dssim_list.append(dssim)
        rssim_list.append(rssim)
        enc_time_list.append(enc_time)
        dec_time_list.append(dec_time)

    print(enc_time_list)
    enc_time_list = [enct / 2 for enct in enc_time_list]
    dec_time_list = [dect / 2 for dect in dec_time_list]
    rgb_result = [dir_list, rbpp_list, rpsnr_list, rssim_list, enc_time_list, dec_time_list]
    depth_result = [dir_list, dbpp_list, dpsnr_list, dssim_list, enc_time_list, dec_time_list]

    print_single_result(rgb_result)
    print_single_result(depth_result)
    return rgb_result, depth_result


def get_metrics_from_collected_single_dirs(root, dirs):
    bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list = [], [], [], [], []
    for dir in dirs:
        exp_dirs = os.listdir(os.path.join(root, dir))
        exp_dirs = [d for d in exp_dirs if d.find("test_epoch") != -1]
        exp_dirs.sort()

        if len(exp_dirs) == 0:
            break

        bpp, psnr, ssim, enc_time, dec_time = [0] * 5
        # 定义时间提取的正则表达式模式
        pattern = r"(\d{6}-\d{6})"

        # 提取时间并排序
        exp_dirs = sorted(exp_dirs, key=lambda x: re.search(pattern, x).group(1), reverse=True)
        for expd in exp_dirs:
            try:
                bpp, psnr, ssim, enc_time, dec_time = get_metrics_from_collected_single_dirs(
                    os.path.join(root, dir, expd)
                )
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

    print_single_result((dirs, bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list))
    return [dirs, bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list]


def print_add_metrics(rgb_metrics, depth_metrics, rgb_offset=0, depth_offset=0, modelname="ELIC"):
    # rgb_metrics = [m[rgb_offset : rgb_offset + 4] for m in rgb_metrics]
    # depth_metrics = [m[depth_offset : depth_offset + 4] for m in depth_metrics]

    outstr = f'"{modelname}": ' + "{\n"
    # 打印各自
    outstr += print_united_result_for_draw(rgb_metrics, "r")
    outstr += print_united_result_for_draw(depth_metrics, "d")
    # 打印总共
    outstr += '"total_bpp"' + ": ["
    bpp_total = [r + d for r, d in zip(rgb_metrics[1], depth_metrics[1])]
    for total_bpp in bpp_total:
        outstr += f"{total_bpp:.4f}, "
    outstr += "],\n"
    outstr += "},\n"
    print(outstr)

    rgb_metrics[1] = bpp_total
    depth_metrics[1] = bpp_total
    print_single_result(rgb_metrics)
    print_single_result(depth_metrics)

    return rgb_metrics, depth_metrics


def print_single_result(result):
    dir_list, bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list = result
    print("model\t\t\t bpp\t\t psnr\t\t ssim\t\t enc_time\t dec_time")
    for i, _ in enumerate(bpp_list):
        print(
            f"{dir_list[i]}\t {bpp_list[i]}\t {psnr_list[i]}\t {ssim_list[i]}\t {enc_time_list[i]}\t {dec_time_list[i]}"
        )
    print()


def print_united_result_for_draw(result, mode="r"):
    metric_name = ["bpp", "psnr", "ssim", "enc_t", "dec_t"]
    metric_name = [mode + m for m in metric_name]
    outstr = ""
    for i, metriclist in enumerate(result):
        if i == 0:
            continue
        outstr += f'"{metric_name[i - 1]}"' + ": ["
        for metric in metriclist:
            outstr += f"{metric:.4f}, "
        outstr += "],\n"
    return outstr


if __name__ == "__main__":
    root = "../experiments"
    model_name1 = "nyuv2_depth_ELIC_master-60"
    model_name2 = "nyuv2_depth_ELIC_master-60"
    if model_name1.find("united") != -1:
        dirs = collect_test_dirs(root, model_name1)
        rgb_metrics, depth_metrics = get_metrics_from_collected_united_dirs(root, dirs)
    else:
        rgb_dirs = collect_test_dirs(root, model_name1)
        depth_dirs = collect_test_dirs(root, model_name2)
        rgb_metrics = get_metrics_from_collected_single_dirs(root, rgb_dirs)
        depth_metrics = get_metrics_from_collected_single_dirs(root, depth_dirs)
    print_add_metrics(rgb_metrics, depth_metrics, 0, 0, model_name1)

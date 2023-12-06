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
# 23-09-18 18:08:10.491 - INFO: Epoch:[232] | Avg rBpp: 0.1566178 | Avg dBpp: 0.0115115 | Avg rPSNR: 34.3340977 | Avg dPSNR: 44.4662399 | Avg rMS-SSIM: 0.9712246 | Avg dMS-SSIM: 0.9943522 | Avg Encoding Latency: 0.256585 | Avg Decoding latency: 0.446901
def get_one_metrics(filename):
    with open(filename, encoding="utf8") as file:
        contents = file.readlines()[-1]
        pattern = r"INFO: Epoch:\[\d*?\] \| Avg rBpp: (\d+\.\d*) \| Avg dBpp: (\d+\.\d*) \| Avg rPSNR: (\d+\.\d*) \| Avg dPSNR: (\d+\.\d*) \| Avg rMS-SSIM: (\d+\.\d*) \| Avg dMS-SSIM: (\d+\.\d*) \| Avg Encoding Latency: (\d+\.\d*) \| Avg Decoding latency: (\d+\.\d*)"
        res = re.findall(pattern, contents)
        res = [float(r.strip()) for r in res[0]]
    return res


def get_all_metrics(root, dirs):
    rbpp_list, dbpp_list, rpsnr_list, dpsnr_list, rssim_list, dssim_list, enc_time_list, dec_time_list = [], [], [], [], [], [], [], []
    # print(dirs)
    dir_list = []
    for dir in dirs:
        exp_dirs = os.listdir(os.path.join(root, dir))
        exp_dirs = [d for d in exp_dirs if d.find("test_epoch") != -1]  # 只取最新的log
        exp_dirs.sort()
        # print(exp_dirs)

        # 防止有的没有eval
        if len(exp_dirs) == 0:
            print(exp_dirs, "not have been tested")
            continue

        rbpp, dbpp, rpsnr, dpsnr, rssim, dssim, enc_time, dec_time = [0] * 8
        # exp_dirs.reverse()

        # 定义时间提取的正则表达式模式
        pattern = r"(\d{6}-\d{6})"

        # 提取时间并排序
        exp_dirs = sorted(exp_dirs, key=lambda x: re.search(pattern, x).group(1), reverse=True)
        last_psnr = 0
        last_ssim = 0
        for expd in exp_dirs:
            try:
                rbpp, dbpp, rpsnr, dpsnr, rssim, dssim, enc_time, dec_time = get_one_metrics(os.path.join(root, dir, expd))
                print(expd)
                if last_psnr != rpsnr or last_ssim != rssim:
                    last_psnr = rpsnr
                    last_ssim = rssim
                else:
                    if os.path.exists(os.path.join(root, dir, expd)):
                        os.remove(os.path.join(root, dir, expd))
                break
            except:
                print(expd, "not have been tested")
                if os.path.exists(os.path.join(root, dir, expd)):
                    os.remove(os.path.join(root, dir, expd))
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
        # exit()

    print(enc_time_list)
    enc_time_list = [enct / 2 for enct in enc_time_list]
    dec_time_list = [dect / 2 for dect in dec_time_list]
    rgb_result = [dir_list, rbpp_list, rpsnr_list, rssim_list, enc_time_list, dec_time_list]
    depth_result = [dir_list, dbpp_list, dpsnr_list, dssim_list, enc_time_list, dec_time_list]

    print_result(rgb_result)
    print_result(depth_result)
    return rgb_result, depth_result


def print_result(result):
    dir_list, bpp_list, psnr_list, ssim_list, enc_time_list, dec_time_list = result
    print("model\t\t\t bpp\t\t psnr\t\t ssim\t\t enc_time\t dec_time")
    for i, _ in enumerate(bpp_list):
        print(f"{dir_list[i]}\t {bpp_list[i]}\t {psnr_list[i]}\t {ssim_list[i]}\t {enc_time_list[i]}\t {dec_time_list[i]}")
    print()


def get_total_metrics(rgb_metrics, depth_metrics, rgb_offset=0, depth_offset=0, modelname="ELIC"):
    # rgb_metrics = [m[rgb_offset : rgb_offset + 4] for m in rgb_metrics]
    # depth_metrics = [m[depth_offset : depth_offset + 4] for m in depth_metrics]

    outstr = f'"{modelname}": ' + "{\n"
    # 打印各自
    outstr += print_result_for_draw(rgb_metrics, "r")
    outstr += print_result_for_draw(depth_metrics, "d")
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
            outstr += f"{metric:.4f}, "
        outstr += "],\n"
    return outstr


def plot_one():
    pass


def get_one_metrics_string(string):
    pattern = r"INFO: Epoch:\[\d*?\] \| Avg rBpp: (\d+\.\d*) \| Avg dBpp: (\d+\.\d*) \| Avg rPSNR: (\d+\.\d*) \| Avg dPSNR: (\d+\.\d*) \| Avg rMS-SSIM: (\d+\.\d*) \| Avg dMS-SSIM: (\d+\.\d*) \| Avg Encoding Latency: (\d+\.\d*) \| Avg Decoding latency: (\d+\.\d*)"
    res = re.findall(pattern, string)
    res = [float(r.strip()) for r in res[0]]
    return res


def print_list(plist, name):
    return name + "=" + str(plist) + "\n"


def get_all_metrics_string(filename):
    rbpp_list, dbpp_list, rpsnr_list, dpsnr_list, rssim_list, dssim_list, enc_time_list, dec_time_list = [], [], [], [], [], [], [], []
    with open(filename) as file:
        contents = file.readlines()
        for c in contents:
            rbpp, dbpp, rpsnr, dpsnr, rssim, dssim, enc_time, dec_time = get_one_metrics_string(c)

        dpsnr_list.append(dpsnr)
        rpsnr_list.append(rpsnr)
        dbpp_list.append(dbpp)
        rbpp_list.append(rbpp)
        dssim_list.append(dssim)
        rssim_list.append(rssim)
        enc_time_list.append(enc_time)
        dec_time_list.append(dec_time)
        outstr = (
            '"padding-mode:{\n"'
            + print_list(rbpp_list)
            + print_list(dbpp_list)
            + print_list(rpsnr_list)
            + print_list(dpsnr_list)
            + print_list(rssim_list)
            + print_list(dssim_list)
            + print_list(enc_time_list)
            + print_list(dec_time_list)
            + print_list([a + b for a, b in zip(rbpp, dbpp)])
        )
        outstr += "}\n"


if __name__ == "__main__":
    root = "/data/chenminghui/ELIC/experiments"
    # 主要是为了方便仅仅打印模型名称
    # model_name = "ELIC_united_LH"
    # model_name = "ELIC_cpf400"
    # model_name = "ELIC_EEM_SE_res"
    # model_name = "ELIC_united4_SE_res"
    # sunrgbd
    # model_name = "ELIC_united_lh"
    # model_name = ".75"
    model_name = "ELIC_united-1"

    dirs = get_dir_list(root, model_name)
    print(dirs)

    rgb_metrics, depth_metrics = get_all_metrics(root, dirs)

    get_total_metrics(rgb_metrics, depth_metrics, 0, 0, model_name)

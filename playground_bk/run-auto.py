import os
import random
import re
import time

import psutil
from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlDeviceGetTemperature, nvmlInit, nvmlShutdown


def get_free_gpu(using_rate=0.9, show_info=False):
    free_gpus = []
    # 初始化
    nvmlInit()
    # 获取GPU个数
    deviceCount = nvmlDeviceGetCount()
    # 总显存
    total_memory = 0
    # 未用总显存
    total_free = 0
    # 已用总显存
    total_used = 0
    # 遍历查看每一个GPU的情况
    for i in range(deviceCount):
        # 创建句柄
        handle = nvmlDeviceGetHandleByIndex(i)
        # 获取信息
        info = nvmlDeviceGetMemoryInfo(handle)
        # 获取gpu名称
        gpu_name = nvmlDeviceGetName(handle)
        # 查看型号、显存、温度、电源
        if show_info:
            print("[ GPU{}: {}".format(i, gpu_name), end="    ")
            print("总共显存: {}G".format((info.total // 1048576) / 1024), end="    ")
            print("空余显存: {}G".format((info.free // 1048576) / 1024), end="    ")
            print("已用显存: {}G".format((info.used // 1048576) / 1024), end="    ")
            print("显存占用率: {:.2%}".format(info.used / info.total), end="    ")
            print("运行温度: {}摄氏度 ]".format(nvmlDeviceGetTemperature(handle, 0)))

        total_memory += (info.total // 1048576) / 1024
        total_free += (info.free // 1048576) / 1024
        total_used += (info.used // 1048576) / 1024
        if info.free / info.total >= using_rate:
            # print(f"GPU {i} is free")f
            free_gpus.append(i)
    # 打印所有GPU信息
    if show_info:
        print("显卡名称：[{}]，显卡数量：[{}]，总共显存；[{}G]，空余显存：[{}G]，已用显存：[{}G]，显存占用率：[{:.2%}]。".format(gpu_name, deviceCount, total_memory, total_free, total_used, (total_used / total_memory)))
        print(free_gpus)

    # 关闭管理工具
    nvmlShutdown()
    return free_gpus


def get_command(file):
    with open(file) as f:
        cmds = f.readlines()
        commands = []
        for cmd in cmds:
            if cmd.find("python") != -1 and cmd.find("#") == -1:
                commands.append(cmd)
    for cmd in commands:
        print(cmd)
    return commands


# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train2united.py -m ELIC_EEM --epochs 400 -lr 1e-4 --save --git --gpu_id 0,1 -q 2_3 --restore --dist&
def every_x_minutes(x=5, sh_file="train_nyuv2_united.sh"):
    cmds = get_command(sh_file)
    cmd_num = 0
    while True:
        # 处理
        free_gpus = get_free_gpu()
        if len(free_gpus) >= 2:
            gpustr = f"{free_gpus[0]},{free_gpus[1]}"
            cmd = re.sub(r"\d,\d", gpustr, cmds[cmd_num])
            cmd = re.sub(r"--master_port \d*", "--master_port " + str(random.randint(6006, 8008)), cmd)
            print(cmd)
            os.system(cmd)
            cmd_num += 1

        # 等待
        time.sleep(60 * x)  # 等待5分钟
        # time.sleep(1)  # 等待1秒钟

        # # 跳出
        # if cmd_num == 3:
        #     print(cmd_num)
        #     break


# for test or single gpu train
def every_x_minutes_test(x=5, sh_file="test_nyuv2.sh"):
    cmds = get_command(sh_file)
    cmd_num = 0
    while True:
        # 处理
        free_gpus = get_free_gpu(0.9)
        if len(free_gpus) >= 1:
            gpustr = f"--gpu_id {free_gpus[0]}"
            cmd = re.sub(r"--gpu_id \d", gpustr, cmds[cmd_num])
            # cmd = re.sub(r"--master_port \d*", "--master_port " + str(random.randint(6006, 8008)), cmd)
            print(cmd)
            os.system(cmd)
            cmd_num += 1

        # 等待
        time.sleep(x * 60)  # 等待5分钟
        # time.sleep(1)  # 等待1秒钟

        # # 跳出
        # if cmd_num == 3:
        #     print(cmd_num)
        #     break


def set_free_cpu(rate=0.1, need_cpu=15):
    cpuinfo = psutil.cpu_percent(interval=0.5, percpu=True)
    freecpu = []
    for i, cinfo in enumerate(cpuinfo):
        if cinfo > rate:
            continue
        freecpu.append(i)
    os.sched_setaffinity(os.getpid(), freecpu[:need_cpu])


if __name__ == "__main__":
    # every_x_minutes(10)
    # every_x_minutes_test(x=10, sh_file="train_nyuv2.sh")
    every_x_minutes_test(x=10, sh_file="train_sunrgbd.sh")

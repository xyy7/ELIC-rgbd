import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class nyuv2(Dataset):
    def __init__(self, train_dir, is_train, is_master=False):
        # self.data_dir = data_dir
        self.image_size = 256

        rgb_dir = train_dir + "/color/*"
        depth_dir = train_dir + "/gt/*"

        self.rgb_files = sorted(glob.glob(rgb_dir))
        self.depth_files = sorted(glob.glob(depth_dir))

        self.train = is_train
        self.is_master = is_master

    def __getitem__(self, index):
        rgb_path = self.rgb_files[index]
        img = Image.open(rgb_path).convert("RGB")
        img = np.array(img) / 255

        img = img.transpose(2, 0, 1)

        depth_path = self.depth_files[index]

        # 注意是255还是10000
        if self.train:
            depth = Image.open(depth_path).convert("L")
            depth = np.array(depth) / 255  #
        else:
            depth = Image.open(depth_path)
            depth = np.array(depth) / 255

        rgb_t = torch.from_numpy(img)  # [3,H,W]

        depth_t = torch.from_numpy(depth)  # [H,W]
        depth_t = torch.unsqueeze(depth_t, 0)

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(rgb_t, output_size=(256, 256))  # 随机裁剪，那就意味着白边其实也有参与训练,只是可能训练比较少
            rgb_t = TF.crop(rgb_t, i, j, h, w)
            depth_t = TF.crop(depth_t, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                rgb_t = TF.hflip(rgb_t)
                depth_t = TF.hflip(depth_t)

            # Random vertical flipping
            if random.random() > 0.5:
                rgb_t = TF.vflip(rgb_t)
                depth_t = TF.vflip(depth_t)
        else:
            if not self.is_master:
                crop_size = (448, 576)
            else:
                crop_size = (256, 256)
            transform = transforms.CenterCrop(crop_size)
            rgb_t = transform(rgb_t)
            depth_t = transform(depth_t)

        rgb_t = rgb_t.type(torch.FloatTensor)
        depth_t = depth_t.type(torch.FloatTensor)

        return rgb_t, depth_t

    def __len__(self):
        return len(self.rgb_files)


class sun(Dataset):
    def __init__(self, train_dir, is_train):
        # self.data_dir = data_dir
        self.image_size = 256

        rgb_dir = train_dir + "/color/*"
        depth_dir = train_dir + "/gt/*"

        self.rgb_files = sorted(glob.glob(rgb_dir))
        self.depth_files = sorted(glob.glob(depth_dir))

        self.train = is_train

    def __getitem__(self, index):
        rgb_path = self.rgb_files[index]
        img = Image.open(rgb_path).convert("RGB")
        img = np.array(img) / 255

        img = img.transpose(2, 0, 1)

        depth_path = self.depth_files[index]

        # 注意是255还是10000
        if self.train:
            depth = Image.open(depth_path)  # .convert('L')
            depth = np.array(depth) / 100000  #
        else:
            depth = Image.open(depth_path)
            depth = np.array(depth) / 100000

        rgb_t = torch.from_numpy(img)  # [3,H,W]

        depth_t = torch.from_numpy(depth)  # [H,W]
        depth_t = torch.unsqueeze(depth_t, 0)

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(rgb_t, output_size=(256, 256))
            rgb_t = TF.crop(rgb_t, i, j, h, w)
            depth_t = TF.crop(depth_t, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                rgb_t = TF.hflip(rgb_t)
                depth_t = TF.hflip(depth_t)

            # Random vertical flipping
            if random.random() > 0.5:
                rgb_t = TF.vflip(rgb_t)
                depth_t = TF.vflip(depth_t)
        else:
            transform = transforms.CenterCrop((448, 576))
            rgb_t = transform(rgb_t)
            depth_t = transform(depth_t)

        rgb_t = rgb_t.type(torch.FloatTensor)
        depth_t = depth_t.type(torch.FloatTensor)

        return rgb_t, depth_t

    def __len__(self):
        return len(self.rgb_files)

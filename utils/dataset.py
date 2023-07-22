from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# nyuv2处理
class ImageFolder(Dataset):

    def __init__(self,
                 root='/data/chenminghui/nyud/nyu5k/nyuv2/test',
                 transform=None,
                 split="depth"):
        splitdir = Path(root) / split
        self.split = split
        print(splitdir)
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        if split == 'rgb':
            self.mode = 'RGB'
            # self.samples = ['/data/chenminghui/nyud/nyu5k/nyuv2/test/rgb/0009.png']
        elif split == 'depth':
            self.mode = 'L'
            # self.samples = ['/data/chenminghui/nyud/nyu5k/nyuv2/test/depth/0009.png']

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.samples.sort()  # 保证rgb能够根据index拿到相应的照片

        self.transform = transform

    def __getitem__(self, index):
        imgname = str(self.samples[index])
        print(imgname)
        img = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize
        if self.mode == 'RGB':
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).float()
            img /= 255.
        elif self.mode == 'L':
            img = np.expand_dims(img, 0).astype('float32')
            img = torch.from_numpy(img).float()
            img /= 10000.

        return img

    def __len__(self):
        return len(self.samples)


# nyuv2处理
class ImageFolderUnited(Dataset):

    def __init__(
        self,
        root='/data/chenminghui/nyud/nyu5k/nyuv2/test',
        transform=None,
    ):
        self.rgb_dataloader = ImageFolder(root=root,
                                          transform=transform,
                                          split='rgb')
        self.depth_dataloader = ImageFolder(root=root,
                                            transform=transform,
                                            split='depth')

    def __getitem__(self, index):
        rgb = self.rgb_dataloader.__getitem__(index)
        depth = self.depth_dataloader.__getitem__(index)

        return rgb, depth

    def __len__(self):
        return len(self.rgb_dataloader)

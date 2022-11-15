import torch
import torchvision

from torchvision import transforms

import numpy as np

import os

from PIL import Image

import math


class tinyImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train, ratio=4, transform=None, target_transform=None, lr_cls=False):
        self.train = train

        self.path_classes = torchvision.datasets.ImageFolder(root=root)
        self.transform = transform
        self.target_transform = target_transform

        self.ratio = ratio

        self.lr_cls = lr_cls

    def __getitem__(self, index):

        hr_img, lbl = self.path_classes[index]

        lr_img = hr_img.resize((hr_img.width//self.ratio, hr_img.height//self.ratio))
        
        # Pad lr image to the same size with hr image for lr image classification
        if self.lr_cls:
            lr_pad = Image.new('RGB', (hr_img.width, hr_img.height))
            lr_pad.paste(lr_img)
            lr_img = lr_pad

        if self.transform is not None:
            lr_img, hr_img = self.transform(lr_img, hr_img)
        

        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
        

        return lr_img, hr_img, lbl
    
    def __len__(self):
        return len(self.path_classes)
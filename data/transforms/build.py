import os
from os.path import join
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def build_transforms(cfg, split="train"):
    assert split in ["train", "test"], "split must be train or test"
    if split == "train":
        transform = A.Compose(
            [
                A.Resize(cfg.INPUT.SIZE_TRAIN, cfg.INPUT.SIZE_TRAIN),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        transform = A.Compose(
            [
                A.Resize(cfg.INPUT.SIZE_TEST, cfg.INPUT.SIZE_TEST), 
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
                ToTensorV2()]
        )
    return transform
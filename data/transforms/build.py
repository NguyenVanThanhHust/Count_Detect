import os
from os.path import join
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
import albumentations.augmentations.functional as AAF

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def build_transforms(cfg, split="train"):
    assert split in ["train", "test"], "split must be train or test"
    if split == "train":
        transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                ToTensorV2(),
            ],  bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']), 
        )
    else:
        transform = A.Compose(
            [
                A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD), 
                ToTensorV2()],
                bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        )
    return transform
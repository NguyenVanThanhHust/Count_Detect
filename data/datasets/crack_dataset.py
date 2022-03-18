import os
from os.path import join
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class CrackDataset(Dataset):
    """
    Crack dataset adapt from https://github.com/khanhha/crack_segmentation/blob/master/data_loader.py
    """
    def __init__(self, data_folder, split="train", transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.img_folder = join(self.data_folder, split, "images") 
        self.mask_folder = join(self.data_folder, split, "masks") 
        self.img_names = next(os.walk(self.img_folder))[2]
        self.transform = transform
    
    def preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask = mask / 255.0
        mask = mask.astype(np.int64)
        return mask

    def __len__(self, ):
        return len(self.img_names)

    def __getitem__(self, idx):
        file_name = self.img_names[idx]
        img_path = join(self.img_folder, file_name)
        mask_path = join(self.mask_folder, file_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = self.preprocess_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

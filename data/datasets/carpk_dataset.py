import os
from os.path import join
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class CarPK_Dataset(Dataset):
    def __init__(self, data_folder, split="train", transforms=None):
        self.data_folder = data_folder
        self.anno_folder = join(data_folder, "Annotations")
        self.img_folder = join(data_folder, "Images")
        self.image_set_file = join(data_folder, "ImageSets", split+".txt")
        with open(self.image_set_file, "r") as handle:
            lines = handle.readlines()
        self.list_images = [l.rstrip() for l in lines]
        self.transforms = transforms
        
    def __len__(self, ):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_path = join(self.img_folder, self.list_images[idx]+".png")
        gt_path = join(self.anno_folder, self.list_images[idx]+".txt")
        with open(gt_path, "r") as handle:
            lines = handle.readlines()
            lines = [l.rstrip() for l in lines]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        box_xyxy = []
        for l in lines:
            coor = l.split(" ")
            coor = [int(c) for c in coor]
            box_xyxy.append(coor[:4])

        box_xyxy = np.array(box_xyxy, dtype=np.float32)

        num_objs = len(box_xyxy)
        labels = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        boxes = np.zeros(box_xyxy.shape, dtype=np.float32)
        boxes[:, :2] = box_xyxy[:, :2]
        boxes[:, 2] = box_xyxy[:, 2] - box_xyxy[:, 0] 
        boxes[:, 3] = box_xyxy[:, 3] - box_xyxy[:, 1]

        if self.transforms is not None:
            results= self.transforms(image=image, bboxes=boxes, category_ids=labels)
        image = results["image"]
        bboxes = results["bboxes"]
        category_ids = results["category_ids"]
        targets = {
            "id": torch.from_numpy(np.array(idx)), 
            "bboxes": torch.tensor(bboxes), 
            "category_ids": torch.tensor(category_ids), 
            "shape": torch.from_numpy(np.array([height, width]))
        }
        return image, targets


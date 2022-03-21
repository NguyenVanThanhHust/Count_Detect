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

    @staticmethod
    def coco_transform(image, bboxes):
        min_side=608
        max_side=1024
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(cols*scale), int(rows*scale)))

        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        new_bboxes = bboxes * scale
        new_h, new_w, _ = new_image.shape
        new_bboxes[:, 0] = np.clip(new_bboxes[:, 0], 0, new_w)
        new_bboxes[:, 1] = np.clip(new_bboxes[:, 1], 0, new_h)
        new_bboxes[:, 2] = np.clip(new_bboxes[:, 2], 0, new_w)
        new_bboxes[:, 3] = np.clip(new_bboxes[:, 3], 0, new_h)

        return new_image, new_bboxes, scale


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
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        # image, boxes, scale = self.coco_transform(image, box_xyxy)
        
        boxes = box_xyxy
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


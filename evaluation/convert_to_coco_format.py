import os
from os.path import join
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import skimage
import json

class CarPK_Dataset(Dataset):
    def __init__(self, data_folder, split="train", transforms=None):
        self.data_folder = data_folder
        self.anno_folder = join(data_folder, "Annotations")
        self.img_folder = join(data_folder, "Images")
        self.image_set_file = join(data_folder, "ImageSets", split+".txt")
        with open(self.image_set_file, "r") as handle:
            lines = handle.readlines()
        self.list_images = [l.rstrip() for l in lines]
        
    def __len__(self, ):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_path = join(self.img_folder, self.list_images[idx]+".png")
        gt_path = join(self.anno_folder, self.list_images[idx]+".txt")
        with open(gt_path, "r") as handle:
            lines = handle.readlines()
            lines = [l.rstrip() for l in lines]
        img = cv2.imread(img_path)
        height, width, _ = img.shape

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
        boxes[:, 2] = boxes[:, 2] -boxes[:, 0]  
        boxes[:, 3] = boxes[:, 3] -boxes[:, 1]  
        return self.list_images[idx]+".png", idx, boxes, (height, width)

data_folder = "../carpk/datasets/CARPK_devkit/data"
datasets = CarPK_Dataset(data_folder=data_folder, split="test")

gts = dict()
gts["categories"] = [{'name': 'fg', 'id': 1}]
gts["images"] = list()
gts["annotations"] = list()

anno_id = 1
for i in range(len(datasets)):
    img_name, idx, boxes, (height, width) = datasets.__getitem__(i)
    image_id = idx + 1
    for box in boxes:
        x, y, w, h = box
        anno = {
            "id": anno_id, 
            "image_id": image_id, 
            "area": int(w*h), 
            "bbox": [int(x), int(y), int(w), int(h)], 
            "category_id": 1,
            "iscrowd": 0
        }
        gts["annotations"].append(anno)
        anno_id += 1
    img_info = {
        "id": image_id, 
        "height": height,
        "width": width,
        "file_name": img_name,
    }
    gts["images"].append(img_info)

with open("gts.json", "w") as handle:
    json.dump(gts, handle)
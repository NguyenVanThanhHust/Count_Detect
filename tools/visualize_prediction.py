import os
from os.path import join
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import io
import contextlib
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

class CarPK_Dataset(Dataset):
    def __init__(self, data_folder, split="train", pred_json_file=None):

        self.data_folder = data_folder
        self.anno_folder = join(data_folder, "Annotations")
        self.img_folder = join(data_folder, "Images")
        self.image_set_file = join(data_folder, "ImageSets", split+".txt")
        with open(self.image_set_file, "r") as handle:
            lines = handle.readlines()
        self.list_images = [l.rstrip() for l in lines]

        pred_json_file = PathManager.get_local_path(pred_json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.pred_coco = COCO(pred_json_file)

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
        boxes = box_xyxy

        img_id = idx + 1
        anno_ids = self.pred_coco.getAnnIds([img_id])
        pred_annos = self.pred_coco.loadAnns(anno_ids)
        pred_boxes = [p["bbox"] for p in pred_annos]
        img = cv2.imread(img_path)
        for box in pred_boxes:
            x, y, w, h = box
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        output_path = os.path.join("./outputs/visualize/vis_res/", self.list_images[idx]+".jpg")
        cv2.imwrite(output_path, img)
        return 

dataset = CarPK_Dataset(data_folder="../carpk/datasets/CARPK_devkit/data", 
                        split="test",
                        pred_json_file="./outputs/predictions.json")

for i in range(dataset.__len__()):
    _ = dataset.__getitem__(i)

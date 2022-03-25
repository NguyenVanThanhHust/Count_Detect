import os
from os.path import join, isfile
import numpy as np
import cv2
import pandas as pd 
import json 

from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class SKU100K_Dataset(Dataset):
    def __init__(self, data_folder, split="train", transforms=None):
        self.data_folder = data_folder
        self.img_folder = join(data_folder, "images")
        self.anno_file = join(data_folder, "annotations", "annotations_"+split+".csv")
        self.transforms = transforms
        self.json_anno_file = join(data_folder, "annotations", "annotations_json_"+split+".json")
        if not isfile(self.json_anno_file):
            print("creating json format gt")
            df = pd.read_csv(self.anno_file)
            df = df.values.tolist()
            self.gts = dict()
            self.gts["categories"] = [{'name': 'fg', 'id': 1}]
            self.gts["images"] = list()
            self.gts["annotations"] = list()
            current_img_name = "None"
            anno_id = 1
            image_id = 0
            for each_anno in df:
                image_name,x1,y1,x2,y2,class_name,image_width,image_height = each_anno
                if image_name !=  current_img_name:
                    image_id += 1
                    current_img_name = image_name
                    img_info = {
                        "id": image_id, 
                        "height": image_height,
                        "width": image_width,
                        "file_name": image_name,
                        }
                    self.gts["images"].append(img_info)
                anno = {
                    "id": anno_id, 
                    "image_id": image_id, 
                    "bbox": [int(x1), int(y1), int(x2), int(y2)], 
                    "category_id": 0,
                }
                self.gts["annotations"].append(anno)
                anno_id += 1
            with open(self.json_anno_file, "w") as handle:
                json.dump(self.gts, handle)
        
        self.gts = COCO(self.json_anno_file)
        self.im_ids = self.gts.getImgIds()

    def __len__(self, ):
        return len(self.im_ids)

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
        new_bboxes[:, 0] = np.clip(new_bboxes[:, 0], 0, new_w-1)
        new_bboxes[:, 1] = np.clip(new_bboxes[:, 1], 0, new_h-1)
        new_bboxes[:, 2] = np.clip(new_bboxes[:, 2], 0, new_w-1)
        new_bboxes[:, 3] = np.clip(new_bboxes[:, 3], 0, new_h-1)

        return new_image, new_bboxes, scale


    def __getitem__(self, idx):
        img_info = self.gts.loadImgs([idx+1])
        ann_ids = self.gts.getAnnIds([idx+1])
        anns = self.gts.loadAnns(ann_ids)
        img_name = img_info[0]["file_name"]

        img_path = join(self.img_folder, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        box_xyxy = []
        for ann in anns:
            x1, y1, x2, y2 = ann["bbox"]
            box_xyxy.append([x1, y1, min(x2, width), min(y2, height)])

        box_xyxy = np.array(box_xyxy, dtype=np.float32)

        num_objs = len(box_xyxy)
        labels = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        # image, boxes, scale = self.coco_transform(image, box_xyxy)
        
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


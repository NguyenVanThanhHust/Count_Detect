import os
import json

import logging
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.ops import nms

from torchvision import transforms

class LitModel(pl.LightningModule):
    def __init__(self, model, optim, thresh=0.5):
        super().__init__()
        self.model = model
        self.optim = optim

        self.thresh = thresh

        self.predictions = dict()
        self.predictions["categories"] = [{'name': 'fg', 'id': 1}]
        self.predictions["images"] = list()
        self.predictions["annotations"] = list()
        self.anno_id = 1

    def forward(self, x):
        pred = self.model(x)
        classifications, regressions, anchors = pred

        transformed_anchors = self.regressBoxes(anchors, regressions)
        transformed_anchors = self.clipBoxes(transformed_anchors, x)

        all_scores = []
        all_boxes = []

        for i in range(classifications.shape[0]):
            scores = torch.squeeze(classifications[i, :, :])
            scores_over_thresh = (scores > self.thresh)
            if scores_over_thresh.sum() == 0:
                all_scores.append([])
                all_boxes.append([])
                continue
            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors[i, :, :])
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
            scores = scores[anchors_nms_idx]
            predicted_boxes = anchorBoxes[anchors_nms_idx]

            all_scores.append(scores)
            all_boxes.append(predicted_boxes)

        return all_scores, all_boxes

    def pred_step(self, batch, batch_idx):
        assert False, "not implemented"
        images, targets = batch 
        images = torch.stack(images, dim=0)
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        import pdb; pdb.set_trace()
        losses = self.loss(preds, targets)
        class_loss, reg_loss = losses
        post_processed_preds = self.forward(images)
        ids = [t["id"].item() for t in targets]
        pred_scores, pred_boxes = post_processed_preds
        for each_id, scores, boxes in zip(ids, pred_scores, pred_boxes):
            scores = scores.detach().cpu().numpy()
            boxes = boxes.detach().cpu().numpy()
            for (score, box) in zip(scores, boxes):
                import pdb; pdb.set_trace()
        self.log('test_reg_loss', reg_loss.item(), on_epoch=True)
        self.log('test_class_loss', class_loss.item(), on_epoch=True)
        self.log('test_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return class_loss + reg_loss

    def test_step(self, batch, batch_idx):
        images, targets = batch 
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        batch_size = images.shape[0]
        processed_preds = []
        for i in range(batch_size):
            each_pred = {
                "scores": [],
                "boxes": [], 
                "classes": [], 
            }
            processed_preds.append(each_pred)
        for pred in preds:
            scores, boxes, classes = pred
            for idx, (score, box, each_class) in enumerate(zip(scores, boxes, classes)):
                processed_preds[idx]["scores"].append(score)
                processed_preds[idx]["boxes"].append(box)
                processed_preds[idx]["classes"].append(each_class)
                
        ids = [t["id"].item() for t in targets]
        shapes = [t["shape"].detach().cpu().numpy() for t in targets]
        for each_id, shape, img, pred in zip(ids, shapes, images, processed_preds):
            scores = pred["scores"]
            boxes = pred["boxes"]
            classes = pred["classes"]
            scores = torch.cat(scores, dim=0)
            boxes = torch.cat(boxes, dim=0)
            classes = torch.cat(classes, dim=0)
            indexes = scores > 0
            scores = scores[indexes]
            boxes = boxes[indexes]
            classes = classes[indexes]
            scores = scores.detach().cpu().numpy()
            boxes = boxes.detach().cpu().numpy()
            if len(scores) == 0:
                img_info = {
                    "id": image_id, 
                    "height": 720,
                    "width": 1280,
                    "file_name": "None",
                    }
                self.predictions["images"].append(img_info)
                continue
            c, h, w = img.shape
            ori_h, ori_w = shape
            image_id = each_id + 1
            boxes[:, 0] = boxes[:, 0] * ori_w / w
            boxes[:, 1] = boxes[:, 1] * ori_h / h
            boxes[:, 2] = boxes[:, 2] * ori_w / w
            boxes[:, 3] = boxes[:, 3] * ori_h / h

            for (score, box) in zip(scores, boxes):
                x1, y1, x2, y2 = box
                anno = {
                    "id": self.anno_id, 
                    "image_id": each_id, 
                    "bbox": [int(x1), int(y1), int(x2), int(y2)], 
                    "category_id": 1,
                    "score": float(score), 
                }
                self.anno_id += 1
                self.predictions["annotations"].append(anno)
            img_info = {
                "id": image_id, 
                "height": 720,
                "width": 1280,
                "file_name": "None",
            }
            self.predictions["images"].append(img_info)

        return 

    def test_epoch_end(self, outputs):
        with open("./outputs/predictions.json", "w") as handle:
            json.dump(self.predictions, handle)

    def configure_optimizers(self):
        return self.optim

def do_test(
        model,
        val_loader,
        optimizer,
        cfg, 
    ):

    my_model = LitModel(model, optimizer, thresh=cfg.TEST.THRESHOLD)
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    # ------------
    # testing
    # ------------
    result = trainer.test(model=my_model, test_dataloaders=val_loader)
    # result = trainer.predict(model=my_model, dataloaders=val_loader)
    print(result)
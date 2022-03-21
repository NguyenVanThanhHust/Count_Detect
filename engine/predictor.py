import os
import json

import logging
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.ops import nms

from torchvision import transforms

from layers.utils import BBoxTransform, ClipBoxes

class LitModel(pl.LightningModule):
    def __init__(self, model, loss, optim, thresh=0.5):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim = optim

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.thresh = thresh

        self.predictions = dict()
        self.predictions["predictions"] = list()
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
            anchorBoxes = anchorBoxes[scores_over_thresh[:, 1]]
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
        images = torch.stack(images, dim=0)
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        losses = self.loss(preds, targets)
        class_loss, reg_loss = losses
        post_processed_preds = self.forward(images)
        ids = [t["id"].item() for t in targets]
        pred_scores, pred_boxes = post_processed_preds
        for each_id, scores, boxes in zip(ids, pred_scores, pred_boxes):
            scores = scores.detach().cpu().numpy()
            boxes = boxes.detach().cpu().numpy()
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
                self.predictions["predictions"].append(anno)

        self.log('test_reg_loss', reg_loss.item(), on_epoch=True)
        self.log('test_class_loss', class_loss.item(), on_epoch=True)
        self.log('test_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return class_loss + reg_loss

    def test_epoch_end(self, outputs):
        with open("./outputs/predictions.json", "w") as handle:
            json.dump(self.predictions, handle)

    def configure_optimizers(self):
        return self.optim

def do_test(
        model,
        val_loader,
        optimizer,
        loss_fn,
        cfg, 
    ):

    my_model = LitModel(model, loss_fn, optimizer, thresh=cfg.TEST.THRESHOLD)
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    # ------------
    # testing
    # ------------
    result = trainer.test(model=my_model, test_dataloaders=val_loader)
    # result = trainer.predict(model=my_model, dataloaders=val_loader)
    print(result)
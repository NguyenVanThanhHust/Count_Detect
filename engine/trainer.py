import logging
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

from layers.utils import BBoxTransform, ClipBoxes
from evaluation.metrics import dice, jaccard

class LitUnet(pl.LightningModule):
    def __init__(self, model, loss, optim):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim = optim

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()


    def forward(self, x):
        pred = self.model(x)
        classifications, regressions, anchors = pred

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        finalScores = finalScores.cuda()
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]

    def training_step(self, batch, batch_idx):
        images, targets = batch 
        images = torch.stack(images, dim=0)
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        losses = self.loss(preds, targets)
        class_loss, reg_loss = losses
        self.log('train_reg_loss', reg_loss.item(), on_epoch=True)
        self.log('train_class_loss', class_loss.item(), on_epoch=True)
        self.log('train_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return class_loss + reg_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch 
        images = torch.stack(images, dim=0)
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        preds = self.model(images)
        losses = self.loss(preds, targets)
        class_loss, reg_loss = losses
        self.log('val_reg_loss', reg_loss.item(), on_epoch=True)
        self.log('val_class_loss', class_loss.item(), on_epoch=True)
        self.log('val_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return class_loss + reg_loss
        

    def test_step(self, batch, batch_idx):
        images, targets = batch 
        images = torch.stack(images, dim=0)
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        preds = self.model(images)
        losses = self.loss(preds, targets)
        class_loss, reg_loss = losses
        self.log('test_reg_loss', reg_loss.item(), on_epoch=True)
        self.log('test_class_loss', class_loss.item(), on_epoch=True)
        self.log('test_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return class_loss + reg_loss
        

    def configure_optimizers(self):
        return self.optim


def do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
    ):

    unet = LitUnet(model, loss_fn, optimizer, )
    
    # ------------
    # training
    # ------------
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    trainer.fit(unet, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=val_loader)
    print(result)
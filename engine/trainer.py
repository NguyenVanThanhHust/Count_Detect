import logging
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_lightning.callbacks import ModelCheckpoint

class LitModel(pl.LightningModule):
    def __init__(self, model, optim):
        super().__init__()
        self.model = model
        self.optim = optim

    def forward(self, x):
        pred = self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch 
        targets = [{k: v for k, v in t.items()} for t in targets]
        losses = self.model(images, targets)
        class_loss, reg_loss = losses
        self.log('train_reg_loss', reg_loss.item(), on_epoch=True)
        self.log('train_class_loss', class_loss.item(), on_epoch=True)
        self.log('train_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return reg_loss + class_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        # class_loss, reg_loss = losses
        # self.log('val_reg_loss', reg_loss.item(), on_epoch=True)
        # self.log('val_class_loss', class_loss.item(), on_epoch=True)
        # self.log('val_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return 
    
    def test_step(self, batch, batch_idx):
        images, targets = batch 
        targets = [{k: v for k, v in t.items()} for t in targets]
        preds = self.model(images)
        # losses = self.loss(preds, targets)
        # class_loss, reg_loss = losses
        # self.log('val_reg_loss', reg_loss.item(), on_epoch=True)
        # self.log('val_class_loss', class_loss.item(), on_epoch=True)
        # self.log('val_loss', (class_loss + reg_loss).item(), on_epoch=True)
        return 

    def configure_optimizers(self):
        return self.optim


def do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        cfg, 
    ):

    my_model = LitModel(model, optimizer,)
    
    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.OUTPUT_DIR,
                                            filename='{epoch}-{train_cls_loss:.2f}-{train_reg_loss:.2f}-{train_loss:.2f}',
                                            every_n_epochs=1 )
    trainer = pl.Trainer(devices=1, accelerator="gpu", 
                        gradient_clip_val=0.5, 
                        max_epochs=cfg.SOLVER.MAX_EPOCHS, 
                        # , overfit_batches=0.01
                        callbacks=[checkpoint_callback], 
                        )
    trainer.fit(my_model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=val_loader)
    print(result)


def do_test(
        model,
        val_loader,
        optimizer,
    ):

    my_model = LitModel(model, loss_fn, optimizer, )
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    # ------------
    # testing
    # ------------
    result = trainer.test(model=my_model, test_dataloaders=val_loader)
    print(result)
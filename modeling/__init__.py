from .retinanet import resnet50
from .loss import FocalLoss

def build_model(cfg):
    model = resnet50(num_classes=cfg.MODEL.NUM_CLASSES, pretrained=True)
    return model

def build_loss(cfg):
    return FocalLoss()
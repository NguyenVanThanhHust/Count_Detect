from .retinanet import RetinaNet
from .utils import convert_fixedbn_model

def build_model(cfg):
    model = RetinaNet(cfg, )
    if cfg.MODEL.WEIGHT != None:
        model.initialize(cfg.MODEL.WEIGHT)
    model = convert_fixedbn_model(model)
    return model
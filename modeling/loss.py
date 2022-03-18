import torch.nn as nn
import torch.nn.functional as F


def bce_loss(output, target):
    return nn.BCEWithLogitsLoss()(output, target.unsqueeze(1).float())
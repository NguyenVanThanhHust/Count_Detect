import torch 

def dice(output, target):
    iou = (2 * (target * output.squeeze(1)).sum() + 1e-15) / (target.sum() + output.squeeze(1).sum() + 1e-15)
    return iou.item()

def jaccard(output, target):
    intersection = (target * output.squeeze(1)).sum()
    union = target.sum() + output.squeeze(1).sum() - intersection
    value = (intersection + 1e-15) / (union + 1e-15) 
    return value.item()

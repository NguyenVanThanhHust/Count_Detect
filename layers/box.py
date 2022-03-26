import numpy as np
import torch
import torch.nn as nn


def generate_anchors(stride, ratio_vals, scales_vals, angles_vals=None):
    'Generate anchors coordinates from scales/ratios'

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return torch.cat([xy1, xy2], dim=1)


def box2delta(boxes, anchors):
    'Convert boxes to deltas from anchors'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    return torch.cat([
        (boxes_ctr - anchors_ctr) / anchors_wh,
        torch.log(boxes_wh / anchors_wh)
    ], 1)


def delta2box(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = (torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1)
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat([
        clamp(pred_ctr - 0.5 * pred_wh),
        clamp(pred_ctr + 0.5 * pred_wh - 1)
    ], 1)


def snap_to_anchors(boxes, classes, size, stride, anchors, num_classes, device, anchor_ious):
    'Snap target boxes (x, y, w, h) to anchors'

    num_anchors = anchors.size()[0] if anchors is not None else 1
    width, height = (int(size[0] / stride), int(size[1] / stride))

    if boxes.nelement() == 0:
        return (torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 4, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device))

    # boxes, classes = boxes.split(4, dim=1)

    # Generate anchors
    x, y = torch.meshgrid([torch.arange(0, size[i], stride, device=device, dtype=classes.dtype) for i in range(2)])
    xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0)
    anchors = anchors.view(-1, 1, 1, 4).to(dtype=classes.dtype)
    anchors = (xyxy + anchors).contiguous().view(-1, 4)

    # Compute overlap between boxes and anchors
    boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:] - 1], 1)
    xy1 = torch.max(anchors[:, None, :2], boxes[:, :2])
    xy2 = torch.min(anchors[:, None, 2:], boxes[:, 2:])
    inter = torch.prod((xy2 - xy1 + 1).clamp(0), 2)
    boxes_area = torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, 1)
    anchors_area = torch.prod(anchors[:, 2:] - anchors[:, :2] + 1, 1)
    overlap = inter / (anchors_area[:, None] + boxes_area - inter)

    # Keep best box per anchor
    overlap, indices = overlap.max(1)
    box_target = box2delta(boxes[indices], anchors)
    box_target = box_target.view(num_anchors, 1, width, height, 4)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()
    
    depth = torch.ones_like(overlap) * -1
    depth[overlap < anchor_ious[0]] = 0  # background
    depth[overlap >= anchor_ious[1]] = classes[indices][overlap >= anchor_ious[1]].squeeze() + 1  # objects
    depth = depth.view(num_anchors, width, height).transpose(1, 2).contiguous()

    # Generate target classes
    cls_target = torch.zeros((anchors.size()[0], num_classes + 1), device=device, dtype=boxes.dtype)
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[overlap < anchor_ious[0]] = num_classes  # background has no class
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()

    return (cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 4, height, width),
            depth.view(num_anchors, 1, height, width))


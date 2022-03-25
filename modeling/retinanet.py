import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

import layers.backbone as backbones_mod
from layers.box import generate_anchors
from layers.utils import decode
from layers.loss import FocalLoss, SmoothL1Loss

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class RetinaNet(nn.Module):
    def __init__(self,
                cfg,  
                backbones='ResNet50FPN', 
                ratios=[1.0, 2.0, 0.5], 
                scales=[4 * 2 ** (i / 3) for i in range(3)],
                angles=None, 
                rotated_bbox=False, 
                anchor_ious=[0.4, 0.5], 
                ):
        super().__init__()

        if not isinstance(backbones, list):
            backbones = [backbones]

        self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})
        self.name = 'RetinaNet'
        self.unused_modules = []
        for b in backbones: self.unused_modules.extend(getattr(self.backbones, b).features.unused_modules)
        self.exporting = False
        self.rotated_bbox = rotated_bbox
        self.anchor_ious = anchor_ious

        self.ratios = ratios
        self.scales = scales
        self.angles = angles if angles is not None else \
                    [-np.pi / 6, 0, np.pi / 6] if self.rotated_bbox else None
        self.anchors = {}
        self.classes = cfg.MODEL.NUM_CLASSES

        self.threshold_train = cfg.MODEL.RETINANET.TRAIN_SCORE_THRESH
        self.top_n = cfg.MODEL.RETINANET.TOP_N
        self.nms_thresh = cfg.MODEL.RETINANET.TEST_SCORE_THRESH
        self.num_detection = cfg.MODEL.RETINANET.TEST_KEEP

        self.stride = max([b.stride for _, b in self.backbones.items()])

        def make_head(output_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, output_size, 3, padding=1)]
            return nn.Sequential(*layers)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.num_anchors = self.num_anchors if not self.rotated_bbox else (self.num_anchors * len(self.angles))
        self.cls_head = make_head(self.classes * self.num_anchors)
        self.box_head = make_head(4 * self.num_anchors) if not self.rotated_bbox \
                        else make_head(6 * self.num_anchors)  # theta -> cos(theta), sin(theta)


    def __repr__(self):
        return '\n'.join([
            '     model: {}'.format(self.name),
            '  backbone: {}'.format(', '.join([k for k, _ in self.backbones.items()])),
            '   classes: {}, anchors: {}'.format(self.classes, self.num_anchors)
        ])

    def forward(self, x, rotated_bbox=None):
        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))

        # Heads forward pass
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]
        if self.training:
            return self._compute_loss(x, cls_heads, box_heads, targets.float())
            
        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]
        
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            stride = x.shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)

            # Decode and filter boxes
            decoded.append(decode(cls_head.contiguous(), box_head.contiguous(), stride, self.nms_thresh, 
                                self.top_n, self.anchors[stride], self.rotated_bbox))


    def _extract_targets(self, targets, stride, size):
        import pdb; pdb.set_trace()
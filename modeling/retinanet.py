import os
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

import layers.backbone as backbones_mod
from layers.box import generate_anchors, snap_to_anchors
from layers.utils import decode, nms
from layers.loss import FocalLoss, SmoothL1Loss

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
        self.nms_thresh = cfg.MODEL.RETINANET.TEST_SCORE_THRESH

        self.test_keep = cfg.MODEL.RETINANET.TEST_KEEP
        self.train_keep = cfg.MODEL.RETINANET.TRAIN_KEEP

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

        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss(beta=0.11)


    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            if self.rotated_bbox:
                ignored += ['box_head.8.bias', 'box_head.8.weight']
            weights = {k: v for k, v in chk['state_dict'].items() if k not in ignored}
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()
            
        else:
            # Initialize backbone(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)

            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

        # Initialize class head prior
        def initialize_prior(layer):
            pi = 0.01
            b = - math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)

        self.cls_head[-1].apply(initialize_prior)
        if self.rotated_bbox:
            self.box_head[-1].apply(initialize_prior)

    def __repr__(self):
        return '\n'.join([
            '     model: {}'.format(self.name),
            '  backbone: {}'.format(', '.join([k for k, _ in self.backbones.items()])),
            '   classes: {}, anchors: {}'.format(self.classes, self.num_anchors)
        ])

    def forward(self, x, targets=None, rotated_bbox=None):
        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))

        # Heads forward pass
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]

        if self.training:
            losses = self._compute_loss(x, cls_heads, box_heads, targets)
            return losses

        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]
        
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            stride = x.shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)

            # Decode and filter boxes
            decoded.append(decode(cls_head.contiguous(), box_head.contiguous(), stride, self.nms_thresh, 
                                self.train_keep, self.anchors[stride], self.rotated_bbox))

        dets = [nms(*d, self.nms_thresh, self.test_keep) for d in decoded]

        # decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        # results = nms(*decoded, self.nms_thresh, self.test_keep)
        return dets

    def _extract_targets(self, targets, stride, size):
        global generate_anchors, snap_to_anchors
        if self.rotated_bbox:
            generate_anchors = generate_anchors_rotated
            snap_to_anchors = snap_to_anchors_rotated
        cls_target, box_target, depth = [], [], []
        for target in targets:
            bboxes = target["bboxes"]
            category_ids = target["category_ids"].double()
            index = category_ids != -1
            keep_boxes = bboxes[index]
            keep_category_ids = category_ids[index]

            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)

            anchors = self.anchors[stride]
            if not self.rotated_bbox:
                anchors = anchors.to(bboxes.device)

            snapped = snap_to_anchors(keep_boxes, keep_category_ids, [s * stride for s in size[::-1]], stride, 
                                    anchors, self.classes, bboxes.device, self.anchor_ious)
            for l, s in zip((cls_target, box_target, depth), snapped): l.append(s)
        
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def _compute_loss(self, x, cls_heads, box_heads, targets):
        cls_losses, box_losses, fg_targets = [], [], []
        for cls_head, box_head in zip(cls_heads, box_heads):
            size = cls_head.shape[-2:]
            stride = x.shape[-1] / cls_head.shape[-1]

            cls_target, box_target, depth = self._extract_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            cls_head = cls_head.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_head, cls_target)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            box_head = box_head.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_head, box_target)
            box_loss = box_mask * box_loss
            box_losses.append(box_loss.sum())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        return cls_loss, box_loss

    def export(self, ):
        return 


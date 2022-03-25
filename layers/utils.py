import numpy as np
import torch
import torch.nn as nn
from ._C import decode as decode_cuda

def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None, rotated=False):
    'Box Decoding and Filtering'

    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6

    # if torch.cuda.is_available():
    #     return decode_cuda(all_cls_head.float(), all_box_head.float(),
    #         anchors.view(-1).tolist(), stride, threshold, top_n, rotated)

    device = all_cls_head.device
    import pdb; pdb.set_trace()
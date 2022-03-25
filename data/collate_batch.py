# https://github.com/NVIDIA/retinanet-examples/blob/main/odtk/data.py
import torch 

def collate_fn(batch):
    data, targets = zip(*batch)
    max_det = max([len(t["category_ids"]) for t in targets])
    new_targets = []
    for t in targets:
        t["bboxes"] = torch.cat([t["bboxes"], torch.ones((max_det - len(t["bboxes"]), 4)) *-1])
        t["category_ids"] = torch.cat([t["category_ids"], torch.ones((max_det - len(t["bboxes"]))) *-1])
        new_targets.append(t)
    targets = tuple(new_targets)
    
    return tuple(zip(*batch))

# https://github.com/NVIDIA/retinanet-examples/blob/main/odtk/data.py
import torch 
import torch.nn.functional as F

def collate_fn(batch):
    data, targets = zip(*batch)
    max_det = max([len(t["category_ids"]) for t in targets])
    new_targets = []
    for t in targets:
        t["bboxes"] = torch.cat([t["bboxes"], torch.ones((max_det - len(t["bboxes"]), 4)) *-1])
        t["category_ids"] = torch.cat([t["category_ids"], torch.ones(max_det - len(t["category_ids"])) *-1])
        new_targets.append(t)
    targets = tuple(new_targets)

    # Pad data to match max batch dimensions    
    sizes = [d.size()[-2:] for d in data]
    w, h = (max(dim) for dim in zip(*sizes))

    data_stack = []
    for datum in data:
        pw, ph = w - datum.size()[-2], h - datum.size()[-1]
        data_stack.append(
            F.pad(datum, (0, ph, 0, pw)) if max(ph, pw) > 0 else datum)

    data = torch.stack(data_stack)
    return data, targets

from torch.utils import data
from .datasets import CarPK_Dataset, SKU100K_Dataset
from .transforms import build_transforms
from .collate_batch import collate_fn

def build_datasets(name, data_folder, transforms, split="train"):
    if name.startswith("CarPK"):
        datasets = CarPK_Dataset(data_folder=data_folder, split=split, transforms=transforms)
    elif name.startswith("SKU110K"):
        datasets = SKU100K_Dataset(data_folder=data_folder, split=split, transforms=transforms)
    else:
        raise NameError("This dataset is not defined")
        import sys; sys.exit()

    return datasets

def make_data_loader(cfg, split="train"):
    if split=="train":
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    transform = build_transforms(cfg, split=split)
    datasets = build_datasets(cfg.DATASETS.TRAIN, cfg.INPUT.FOLDER, transform, split)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader
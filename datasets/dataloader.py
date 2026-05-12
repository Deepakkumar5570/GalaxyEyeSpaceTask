from torch.utils.data import DataLoader

from datasets.dataset import GalaxEyeDataset

from datasets.transforms import (
    get_train_transforms,
    get_val_transforms
)


def build_dataloaders(config):

    train_dataset = GalaxEyeDataset(
        root_dir=f'{config["DATASET"]["ROOT"]}/train/train',
        config=config,
        transforms=get_train_transforms()
    )

    val_dataset = GalaxEyeDataset(
        root_dir=f'{config["DATASET"]["ROOT"]}/val/val',
        config=config,
        transforms=get_val_transforms()
    )

    test_dataset = GalaxEyeDataset(
        root_dir=f'{config["DATASET"]["ROOT"]}/test/test',
        config=config,
        transforms=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["TRAIN"]["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["TRAIN"]["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["TRAIN"]["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader
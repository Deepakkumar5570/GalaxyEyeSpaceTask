from torch.utils.data import DataLoader

from datasets.dataset import GalaxEyeDataset

from datasets.transforms import (
    get_train_transforms,
    get_val_transforms
)


# =========================================================
# BUILD DATALOADERS
# =========================================================

def build_dataloaders(config):

    # =====================================================
    # TRAIN DATASET
    # =====================================================

    train_dataset = GalaxEyeDataset(

        root_dir=f'{config["DATASET"]["ROOT"]}/train',

        config=config,

        transforms=get_train_transforms()
    )

    # =====================================================
    # VALIDATION DATASET
    # =====================================================

    val_dataset = GalaxEyeDataset(

        root_dir=f'{config["DATASET"]["ROOT"]}/val',

        config=config,

        transforms=get_val_transforms()
    )

    # =====================================================
    # TEST DATASET
    # =====================================================

    test_dataset = GalaxEyeDataset(

        root_dir=f'{config["DATASET"]["ROOT"]}/test',

        config=config,

        transforms=get_val_transforms()
    )

    # =====================================================
    # TRAIN LOADER
    # =====================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=config["TRAIN"]["BATCH_SIZE"],

        shuffle=True,

        num_workers=config["TRAIN"]["NUM_WORKERS"],

        pin_memory=True,

        drop_last=True
    )

    # =====================================================
    # VALIDATION LOADER
    # =====================================================

    val_loader = DataLoader(

        val_dataset,

        batch_size=config["TRAIN"]["BATCH_SIZE"],

        shuffle=False,

        num_workers=config["TRAIN"]["NUM_WORKERS"],

        pin_memory=True
    )

    # =====================================================
    # TEST LOADER
    # =====================================================

    test_loader = DataLoader(

        test_dataset,

        batch_size=config["TRAIN"]["BATCH_SIZE"],

        shuffle=False,

        num_workers=config["TRAIN"]["NUM_WORKERS"],

        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )
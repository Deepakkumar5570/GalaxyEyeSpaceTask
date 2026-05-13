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






















# for kaggle .............................................................

# import os

# from torch.utils.data import DataLoader

# from datasets.dataset import GalaxEyeDataset

# from datasets.transforms import (
#     get_train_transforms,
#     get_val_transforms
# )


# # =========================================================
# # AUTO DATASET PATH DETECTION
# # =========================================================

# def resolve_split_path(root, split):

#     # =====================================================
#     # KAGGLE STYLE
#     # train/train
#     # =====================================================

#     kaggle_path = os.path.join(
#         root,
#         split,
#         split
#     )

#     # =====================================================
#     # LOCAL STYLE
#     # train
#     # =====================================================

#     local_path = os.path.join(
#         root,
#         split
#     )

#     # =====================================================
#     # AUTO DETECT
#     # =====================================================

#     if os.path.exists(kaggle_path):

#         return kaggle_path

#     elif os.path.exists(local_path):

#         return local_path

#     else:

#         raise FileNotFoundError(

#             f\"Dataset split not found:\\n\"

#             f\"Checked:\\n{local_path}\\n{kaggle_path}\"
#         )


# # =========================================================
# # BUILD DATALOADERS
# # =========================================================

# def build_dataloaders(config):

#     root = config["DATASET"]["ROOT"]

#     # =====================================================
#     # AUTO RESOLVE PATHS
#     # =====================================================

#     train_path = resolve_split_path(
#         root,
#         "train"
#     )

#     val_path = resolve_split_path(
#         root,
#         "val"
#     )

#     test_path = resolve_split_path(
#         root,
#         "test"
#     )

#     print(f\"Train Path: {train_path}\")

#     print(f\"Val Path: {val_path}\")

#     print(f\"Test Path: {test_path}\")

#     # =====================================================
#     # DATASETS
#     # =====================================================

#     train_dataset = GalaxEyeDataset(

#         root_dir=train_path,

#         config=config,

#         transforms=get_train_transforms()
#     )

#     val_dataset = GalaxEyeDataset(

#         root_dir=val_path,

#         config=config,

#         transforms=get_val_transforms()
#     )

#     test_dataset = GalaxEyeDataset(

#         root_dir=test_path,

#         config=config,

#         transforms=get_val_transforms()
#     )

#     # =====================================================
#     # DATALOADERS
#     # =====================================================

#     train_loader = DataLoader(

#         train_dataset,

#         batch_size=config["TRAIN"]["BATCH_SIZE"],

#         shuffle=True,

#         num_workers=config["TRAIN"]["NUM_WORKERS"],

#         pin_memory=True,

#         drop_last=True
#     )

#     val_loader = DataLoader(

#         val_dataset,

#         batch_size=config["TRAIN"]["BATCH_SIZE"],

#         shuffle=False,

#         num_workers=config["TRAIN"]["NUM_WORKERS"],

#         pin_memory=True
#     )

#     test_loader = DataLoader(

#         test_dataset,

#         batch_size=config["TRAIN"]["BATCH_SIZE"],

#         shuffle=False,

#         num_workers=config["TRAIN"]["NUM_WORKERS"],

#         pin_memory=True
#     )

#     return (
#         train_loader,
#         val_loader,
#         test_loader
#     )
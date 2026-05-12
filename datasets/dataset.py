import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset


class GalaxEyeDataset(Dataset):

    def __init__(
        self,
        root_dir,
        config,
        transforms=None
    ):

        self.root_dir = root_dir

        self.transforms = transforms

        self.patch_size = config["DATASET"]["PATCH_SIZE"]

        self.mode = config["DATASET"]["MODE"]

        self.pre_path = os.path.join(root_dir, "pre-event")

        self.post_path = os.path.join(root_dir, "post-event")

        self.mask_path = os.path.join(root_dir, "target")

        self.files = sorted(os.listdir(self.pre_path))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        pre = cv2.imread(
            os.path.join(self.pre_path, file_name)
        )

        post = cv2.imread(
            os.path.join(self.post_path, file_name)
        )

        mask = cv2.imread(
            os.path.join(self.mask_path, file_name),
            cv2.IMREAD_GRAYSCALE
        )

        pre = cv2.resize(pre, (self.patch_size, self.patch_size))

        post = cv2.resize(post, (self.patch_size, self.patch_size))

        mask = cv2.resize(mask, (self.patch_size, self.patch_size))

        if self.transforms:

            transformed = self.transforms(
                image=pre,
                image0=post,
                mask=mask
            )

            pre = transformed["image"]

            post = transformed["image0"]

            mask = transformed["mask"]

        pre = torch.tensor(pre).permute(2,0,1).float() / 255.0

        post = torch.tensor(post).permute(2,0,1).float() / 255.0

        mask = torch.tensor((mask > 0).astype(np.int64))

        # CONCAT MODE
        if self.mode == "concatenated":

            image = torch.cat([pre, post], dim=0)

            return image, mask

        # SIAMESE MODE
        else:

            return pre, post, mask
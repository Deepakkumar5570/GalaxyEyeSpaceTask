import os
import cv2
import torch
import logging
import warnings
import numpy as np

from torch.utils.data import Dataset


# =========================================================
# SUPPRESS WARNINGS
# =========================================================

warnings.filterwarnings("ignore")

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

logging.getLogger("PIL").setLevel(logging.ERROR)

try:

    cv2.setNumThreads(0)

except:

    pass

try:

    cv2.utils.logging.setLogLevel(
        cv2.utils.logging.LOG_LEVEL_ERROR
    )

except:

    pass


# =========================================================
# DATASET
# =========================================================

class GalaxEyeDataset(Dataset):

    def __init__(

        self,

        root_dir,

        config,

        transforms=None

    ):

        self.root_dir = root_dir

        self.config = config

        self.transforms = transforms

        self.patch_size = config["DATASET"]["PATCH_SIZE"]

        self.mode = config["DATASET"]["MODE"]

        # =====================================================
        # PATHS
        # =====================================================

        self.pre_path = os.path.join(
            root_dir,
            "pre-event"
        )

        self.post_path = os.path.join(
            root_dir,
            "post-event"
        )

        self.mask_path = os.path.join(
            root_dir,
            "target"
        )

        # =====================================================
        # FILES
        # =====================================================

        self.files = sorted([

            f for f in os.listdir(self.pre_path)

            if f.endswith(

                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff"
                )
            )
        ])

        self.samples = []

        # =====================================================
        # VALIDATE FILES
        # =====================================================

        for file_name in self.files:

            pre_file = os.path.join(
                self.pre_path,
                file_name
            )

            post_file = os.path.join(
                self.post_path,
                file_name
            )

            mask_file = os.path.join(
                self.mask_path,
                file_name
            )

            if (

                os.path.exists(pre_file)

                and os.path.exists(post_file)

                and os.path.exists(mask_file)

            ):

                self.samples.append(

                    (
                        pre_file,
                        post_file,
                        mask_file
                    )
                )

        print(

            f"\nLoaded {len(self.samples)} samples from {root_dir}"

        )

    # =========================================================
    # LENGTH
    # =========================================================

    def __len__(self):

        return len(self.samples)

    # =========================================================
    # SAFE IMAGE READER
    # =========================================================

    def read_image(

        self,

        path,

        flag

    ):

        image = cv2.imdecode(

            np.fromfile(
                path,
                dtype=np.uint8
            ),

            flag
        )

        return image

    # =========================================================
    # GET ITEM
    # =========================================================

    def __getitem__(self, idx):

        pre_file, post_file, mask_file = self.samples[idx]

        # =====================================================
        # READ IMAGES
        # =====================================================

        pre_image = self.read_image(
            pre_file,
            cv2.IMREAD_COLOR
        )

        post_image = self.read_image(
            post_file,
            cv2.IMREAD_COLOR
        )

        mask = self.read_image(
            mask_file,
            cv2.IMREAD_GRAYSCALE
        )

        # =====================================================
        # BGR TO RGB
        # =====================================================

        pre_image = cv2.cvtColor(
            pre_image,
            cv2.COLOR_BGR2RGB
        )

        post_image = cv2.cvtColor(
            post_image,
            cv2.COLOR_BGR2RGB
        )

        # =====================================================
        # RESIZE
        # =====================================================

        pre_image = cv2.resize(

            pre_image,

            (
                self.patch_size,
                self.patch_size
            )
        )

        post_image = cv2.resize(

            post_image,

            (
                self.patch_size,
                self.patch_size
            )
        )

        mask = cv2.resize(

            mask,

            (
                self.patch_size,
                self.patch_size
            ),

            interpolation=cv2.INTER_NEAREST
        )

        # =====================================================
        # NORMALIZE
        # =====================================================

        pre_image = (

            pre_image.astype(np.float32)

            / 255.0
        )

        post_image = (

            post_image.astype(np.float32)

            / 255.0
        )

        mask = (

            mask > 0

        ).astype(np.int64)

        # =====================================================
        # AUGMENTATIONS
        # =====================================================

        if self.transforms is not None:

            augmented = self.transforms(

                image=pre_image,

                image0=post_image,

                mask=mask
            )

            pre_image = augmented["image"]

            post_image = augmented["image0"]

            mask = augmented["mask"]

        # =====================================================
        # TO TENSOR
        # =====================================================

        pre_image = torch.tensor(
            pre_image
        ).permute(2, 0, 1).float()

        post_image = torch.tensor(
            post_image
        ).permute(2, 0, 1).float()

        mask = torch.tensor(
            mask
        ).long()

        # =====================================================
        # SIAMESE MODE
        # =====================================================

        if self.mode == "siamese":

            return (

                pre_image,

                post_image,

                mask
            )

        # =====================================================
        # CONCAT MODE
        # =====================================================

        image = torch.cat(

            [
                pre_image,
                post_image
            ],

            dim=0
        )

        return image, mask
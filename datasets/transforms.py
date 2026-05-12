import albumentations as A


# =========================================================
# TRAIN TRANSFORMS
# =========================================================

def get_train_transforms():

    return A.Compose(

        [

            # =============================================
            # GEOMETRIC
            # =============================================

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=0,
                p=0.5
            ),

            # =============================================
            # PHOTOMETRIC
            # =============================================

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),

            # =============================================
            # NOISE
            # =============================================

            A.GaussNoise(
                std_range=(0.02, 0.08),
                p=0.3
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2
            ),

            # =============================================
            # REGULARIZATION
            # =============================================

            A.CoarseDropout(
                num_holes_range=(2, 6),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.3
            ),

        ],

        additional_targets={
            "image0": "image"
        }

    )


# =========================================================
# VAL / TEST TRANSFORMS
# =========================================================

def get_val_transforms():

    return A.Compose(

        [],

        additional_targets={
            "image0": "image"
        }

    )
import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.config import load_config
from utils.device import get_device

from models.model_factory import build_model


# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True
)

args = parser.parse_args()


# =========================================================
# CONFIG
# =========================================================

config = load_config(args.config)

DEVICE = get_device()


# =========================================================
# MODEL
# =========================================================

model = build_model(config)

checkpoint_path = (
    f'{config["OUTPUT"]["CHECKPOINT_DIR"]}/best_model.pth'
)

state_dict = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)

model.eval()


# =========================================================
# PATHS
# =========================================================

root = f'{config["DATASET"]["ROOT"]}/test/test'

pre_dir = os.path.join(root, "pre-event")

post_dir = os.path.join(root, "post-event")

mask_dir = os.path.join(root, "target")

save_dir = "outputs/visualizations"

os.makedirs(save_dir, exist_ok=True)


# =========================================================
# FILES
# =========================================================

files = sorted(os.listdir(pre_dir))

random_files = random.sample(files, 5)

patch_size = config["DATASET"]["PATCH_SIZE"]


# =========================================================
# LOOP
# =========================================================

for idx, file_name in enumerate(random_files):

    pre_path = os.path.join(pre_dir, file_name)

    post_path = os.path.join(post_dir, file_name)

    mask_path = os.path.join(mask_dir, file_name)

    pre_image = cv2.imread(pre_path)

    post_image = cv2.imread(post_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)

    post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

    pre_image = cv2.resize(pre_image, (patch_size, patch_size))

    post_image = cv2.resize(post_image, (patch_size, patch_size))

    mask = cv2.resize(mask, (patch_size, patch_size))

    pre_tensor = (
        torch.tensor(pre_image)
        .permute(2,0,1)
        .float() / 255.0
    )

    post_tensor = (
        torch.tensor(post_image)
        .permute(2,0,1)
        .float() / 255.0
    )

    pre_tensor = pre_tensor.unsqueeze(0).to(DEVICE)

    post_tensor = post_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        if config["DATASET"]["MODE"] == "siamese":

            logits = model(
                pre_tensor,
                post_tensor
            )

        else:

            image = torch.cat(
                [pre_tensor, post_tensor],
                dim=1
            )

            logits = model(image)

        prediction = torch.argmax(
            logits,
            dim=1
        )

    prediction = (
        prediction.squeeze()
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    axes[0].imshow(pre_image)
    axes[0].set_title("Pre Image")
    axes[0].axis("off")

    axes[1].imshow(post_image)
    axes[1].set_title("Post Image")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    axes[3].imshow(prediction, cmap="gray")
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    save_path = os.path.join(
        save_dir,
        f"prediction_{idx+1}.png"
    )

    plt.savefig(save_path)

    plt.close()

    print(f"Saved: {save_path}")

print("✅ Visualization completed")
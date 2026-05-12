import os
import cv2
import argparse
import numpy as np
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

parser.add_argument(
    "--pre_image",
    type=str,
    required=True
)

parser.add_argument(
    "--post_image",
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

print("✅ Model loaded successfully")


# =========================================================
# LOAD IMAGES
# =========================================================

patch_size = config["DATASET"]["PATCH_SIZE"]

pre_image = cv2.imread(args.pre_image)

post_image = cv2.imread(args.post_image)

pre_image = cv2.cvtColor(
    pre_image,
    cv2.COLOR_BGR2RGB
)

post_image = cv2.cvtColor(
    post_image,
    cv2.COLOR_BGR2RGB
)

pre_image = cv2.resize(
    pre_image,
    (patch_size, patch_size)
)

post_image = cv2.resize(
    post_image,
    (patch_size, patch_size)
)


# =========================================================
# NORMALIZATION
# =========================================================

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


# =========================================================
# INFERENCE
# =========================================================

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
    .astype(np.uint8)
)


# =========================================================
# SAVE PREDICTION
# =========================================================

save_dir = "outputs/predictions"

os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(
    save_dir,
    "prediction.png"
)

cv2.imwrite(
    save_path,
    prediction * 255
)


# =========================================================
# CHANGE PERCENTAGE
# =========================================================

changed_pixels = np.sum(prediction == 1)

total_pixels = prediction.size

change_percent = (
    changed_pixels / total_pixels
) * 100

print(f"Change Percentage: {change_percent:.2f}%")

print(f"Prediction saved at: {save_path}")
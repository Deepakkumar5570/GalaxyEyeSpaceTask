import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.config import load_config
from utils.device import get_device
from models.model_factory import build_model
from metrics.metrics import compute_metrics


def get_checkpoint_path(config):
    checkpoint_cfg = config["OUTPUT"]["CHECKPOINT_DIR"]

    # If the user provided a full checkpoint file path, return it directly.
    # Accept either an existing file path or any string that ends with a common
    # checkpoint extension so configs can point to the file itself.
    if isinstance(checkpoint_cfg, str):
        # Expand user and vars
        checkpoint_cfg = os.path.expanduser(checkpoint_cfg)

        # If it's an existing file, use it
        if os.path.exists(checkpoint_cfg) and os.path.isfile(checkpoint_cfg):
            return checkpoint_cfg

        # If it looks like a checkpoint filename (endswith .pth or .pt),
        # return it as-is (even if it doesn't exist yet) so the caller gets
        # a clear path and a descriptive FileNotFoundError.
        lower = checkpoint_cfg.lower()
        if lower.endswith('.pth') or lower.endswith('.pt'):
            return checkpoint_cfg

    # Otherwise treat value as a directory and append the default filename
    return os.path.join(
        config["OUTPUT"]["CHECKPOINT_DIR"],
        "best_model.pth"
    )


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    return model


def load_model(config_path):
    config = load_config(config_path)
    device = get_device()

    model = build_model(config)
    checkpoint_path = get_checkpoint_path(config)

    model = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    model = model.to(device)
    model.eval()

    return model, config, device


def read_image_as_tensor(path, patch_size, device):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Image not found or cannot be opened: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (patch_size, patch_size)
    )

    image = image.astype(np.float32) / 255.0
    tensor = torch.tensor(image).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0).to(device)

    return tensor


def infer_image_pair(
    model,
    config,
    pre_image_path,
    post_image_path,
    device
):
    patch_size = config["DATASET"]["PATCH_SIZE"]

    pre_tensor = read_image_as_tensor(
        pre_image_path,
        patch_size,
        device
    )

    post_tensor = read_image_as_tensor(
        post_image_path,
        patch_size,
        device
    )

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

    prediction = prediction.squeeze().cpu().numpy().astype(np.uint8)
    return prediction


def evaluate_model(model, loader, device, config):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(loader):
            if config["DATASET"]["MODE"] == "siamese":
                pre_images, post_images, masks = batch
                pre_images = pre_images.to(device)
                post_images = post_images.to(device)
                masks = masks.to(device)
                logits = model(pre_images, post_images)
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)

            metrics = compute_metrics(logits, masks)
            all_metrics.append(metrics)

    if not all_metrics:
        return {"IoU": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    avg_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0]
    }

    return avg_metrics


def save_prediction(prediction, save_dir="outputs/predictions", filename="prediction.png"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, prediction * 255)

    return save_path

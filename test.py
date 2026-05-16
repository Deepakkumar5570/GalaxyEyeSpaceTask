import argparse
import yaml
import torch
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets.dataset import GalaxEyeDataset
from datasets.transforms import get_val_transforms

from models.model_factory import build_model

from metrics.metrics import compute_metrics


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True
)

args = parser.parse_args()

with open(args.config) as f:

    config = yaml.safe_load(f)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


test_dataset = GalaxEyeDataset(
    root_dir=f'{config["DATASET"]["ROOT"]}/test',
    config=config,
    transforms=get_val_transforms()
)


test_loader = DataLoader(
    test_dataset,
    batch_size=config["TRAIN"]["BATCH_SIZE"],
    shuffle=False,
    num_workers=config["TRAIN"]["NUM_WORKERS"]
)


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

all_metrics = []

with torch.no_grad():

    for batch in tqdm(test_loader):

        if config["DATASET"]["MODE"] == "siamese":

            pre, post, masks = batch

            pre = pre.to(DEVICE)
            post = post.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(pre, post)

        else:

            images, masks = batch

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

        metrics = compute_metrics(logits, masks)

        all_metrics.append(metrics)


avg_f1 = np.mean([
    m["F1"] for m in all_metrics
])

print(f"Final Test F1: {avg_f1:.4f}")
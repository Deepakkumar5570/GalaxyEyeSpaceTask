import argparse
import yaml
import torch

from datasets.dataset import GalaxEyeDataset
from datasets.transforms import (
    get_train_transforms,
    get_val_transforms
)

from torch.utils.data import DataLoader

from models.model_factory import build_model

from losses.combined_loss import CombinedLoss

from engine.trainer import Trainer


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


train_dataset = GalaxEyeDataset(
    root_dir=f'{config["DATASET"]["ROOT"]}/train/train',
    config=config,
    transforms=get_train_transforms()
)


train_loader = DataLoader(
    train_dataset,
    batch_size=config["TRAIN"]["BATCH_SIZE"],
    shuffle=True,
    num_workers=config["TRAIN"]["NUM_WORKERS"]
)


model = build_model(config)

model = model.to(DEVICE)

criterion = CombinedLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["TRAIN"]["LR"]
)


trainer = Trainer(
    model,
    criterion,
    optimizer,
    config,
    DEVICE
)


for epoch in range(config["TRAIN"]["EPOCHS"]):

    loss = trainer.train_epoch(train_loader)

    print(f"Epoch {epoch+1} Loss: {loss:.4f}")
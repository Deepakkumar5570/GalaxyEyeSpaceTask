import argparse
import torch
import multiprocessing

from utils.config import load_config
from utils.device import get_device
from utils.seed import seed_everything

from datasets.dataloader import (
    build_dataloaders
)

from models.model_factory import build_model

from losses.combined_loss import CombinedLoss

from engine.trainer import Trainer


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # ARGUMENTS
    # =====================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    # =====================================================
    # CONFIG
    # =====================================================

    config = load_config(args.config)

    # =====================================================
    # SEED
    # =====================================================

    seed_everything(42)

    # =====================================================
    # DEVICE
    # =====================================================

    DEVICE = get_device()

    print(f"Using Device: {DEVICE}")

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader, val_loader, _ = build_dataloaders(
        config
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = build_model(config)

    if torch.cuda.device_count() > 1:

        print(
            f"Using {torch.cuda.device_count()} GPUs"
        )

        model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)

    # =====================================================
    # LOSS
    # =====================================================

    criterion = CombinedLoss()

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config["TRAIN"]["LR"],

        weight_decay=1e-4
    )

    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=config["TRAIN"]["EPOCHS"]
    )

    # =====================================================
    # TRAINER
    # =====================================================

    trainer = Trainer(

        model=model,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        config=config,

        device=DEVICE
    )

    # =====================================================
    # TRAIN
    # =====================================================

    trainer.fit(
        train_loader,
        val_loader
    )


# =========================================================
# WINDOWS SAFE
# =========================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()
import argparse

import numpy as np
from torch.utils.data import DataLoader

from datasets.dataset import GalaxEyeDataset
from datasets.transforms import get_val_transforms
from utils.model_utils import (
    evaluate_model,
    load_model
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    model, config, device = load_model(args.config)
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

    metrics = evaluate_model(
        model,
        test_loader,
        device,
        config
    )

    print(f"Final Test F1: {metrics['F1']:.4f}")
    print(f"Final Test IoU: {metrics['IoU']:.4f}")
    # print(f"Final Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Final Test Precision: {metrics['Precision']:.4f}")
    print(f"Final Test Recall: {metrics['Recall']:.4f}")
   


if __name__ == "__main__":
    main()
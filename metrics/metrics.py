import torch
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)


SMOOTH = 1e-6


def compute_metrics(logits, masks):

    preds = torch.argmax(logits, dim=1)

    preds = preds.detach().cpu().numpy().flatten()

    masks = masks.detach().cpu().numpy().flatten()

    iou = jaccard_score(
        masks,
        preds,
        average="binary"
    )

    precision = precision_score(
        masks,
        preds,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        masks,
        preds,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        masks,
        preds,
        average="binary",
        zero_division=0
    )

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
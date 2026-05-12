import torch
import numpy as np

from tqdm import tqdm

from metrics.metrics import compute_metrics


class Evaluator:

    def __init__(
        self,
        model,
        device,
        config
    ):

        self.model = model

        self.device = device

        self.config = config

    def evaluate(self, loader):

        self.model.eval()

        all_metrics = []

        with torch.no_grad():

            for batch in tqdm(loader):

                if self.config["DATASET"]["MODE"] == "siamese":

                    pre, post, masks = batch

                    pre = pre.to(self.device)
                    post = post.to(self.device)
                    masks = masks.to(self.device)

                    logits = self.model(pre, post)

                else:

                    images, masks = batch

                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    logits = self.model(images)

                metrics = compute_metrics(logits, masks)

                all_metrics.append(metrics)

        avg_iou = np.mean([
            m["IoU"] for m in all_metrics
        ])

        avg_precision = np.mean([
            m["Precision"] for m in all_metrics
        ])

        avg_recall = np.mean([
            m["Recall"] for m in all_metrics
        ])

        avg_f1 = np.mean([
            m["F1"] for m in all_metrics
        ])

        return {
            "IoU": avg_iou,
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1": avg_f1
        }
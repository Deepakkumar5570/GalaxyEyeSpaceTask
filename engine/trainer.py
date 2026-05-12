import os
import torch
import numpy as np

from tqdm import tqdm

from metrics.metrics import compute_metrics

from utils.early_stopping import EarlyStopping


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device
    ):

        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.config = config

        self.device = device

        self.early_stopper = EarlyStopping(
            patience=5,
            mode="max"
        )

    # =====================================================
    # TRAIN ONE EPOCH
    # =====================================================

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        train_bar = tqdm(loader)

        for batch in train_bar:

            # =============================================
            # SIAMESE MODE
            # =============================================

            if self.config["DATASET"]["MODE"] == "siamese":

                pre_images, post_images, masks = batch

                pre_images = pre_images.to(self.device)

                post_images = post_images.to(self.device)

                masks = masks.to(self.device)

                logits = self.model(
                    pre_images,
                    post_images
                )

            # =============================================
            # CONCAT MODE
            # =============================================

            else:

                images, masks = batch

                images = images.to(self.device)

                masks = masks.to(self.device)

                logits = self.model(images)

            # =============================================
            # LOSS
            # =============================================

            loss = self.criterion(
                logits,
                masks
            )

            # =============================================
            # BACKPROP
            # =============================================

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()

            train_bar.set_description(
                f'Train Loss: {loss.item():.4f}'
            )

        avg_loss = total_loss / len(loader)

        return avg_loss

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self, loader):

        self.model.eval()

        total_loss = 0

        all_metrics = []

        with torch.no_grad():

            val_bar = tqdm(loader)

            for batch in val_bar:

                # =========================================
                # SIAMESE MODE
                # =========================================

                if self.config["DATASET"]["MODE"] == "siamese":

                    pre_images, post_images, masks = batch

                    pre_images = pre_images.to(self.device)

                    post_images = post_images.to(self.device)

                    masks = masks.to(self.device)

                    logits = self.model(
                        pre_images,
                        post_images
                    )

                # =========================================
                # CONCAT MODE
                # =========================================

                else:

                    images, masks = batch

                    images = images.to(self.device)

                    masks = masks.to(self.device)

                    logits = self.model(images)

                # =========================================
                # LOSS
                # =========================================

                loss = self.criterion(
                    logits,
                    masks
                )

                total_loss += loss.item()

                metrics = compute_metrics(
                    logits,
                    masks
                )

                all_metrics.append(metrics)

        # =============================================
        # AVERAGE METRICS
        # =============================================

        avg_loss = total_loss / len(loader)

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
            "loss": avg_loss,
            "iou": avg_iou,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

    # =====================================================
    # COMPLETE TRAINING
    # =====================================================

    def fit(
        self,
        train_loader,
        val_loader
    ):

        epochs = self.config["TRAIN"]["EPOCHS"]

        checkpoint_dir = self.config[
            "OUTPUT"
        ]["CHECKPOINT_DIR"]

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

        for epoch in range(epochs):

            print(
                f'\n========== EPOCH {epoch+1}/{epochs} =========='
            )

            # =========================================
            # TRAIN
            # =========================================

            train_loss = self.train_epoch(
                train_loader
            )

            # =========================================
            # VALIDATE
            # =========================================

            metrics = self.validate(
                val_loader
            )

            # =========================================
            # LR SCHEDULER
            # =========================================

            if self.scheduler is not None:

                self.scheduler.step()

            # =========================================
            # PRINT RESULTS
            # =========================================

            print(
                f'Train Loss : {train_loss:.4f}'
            )

            print(
                f'Val Loss   : {metrics["loss"]:.4f}'
            )

            print(
                f'IoU        : {metrics["iou"]:.4f}'
            )

            print(
                f'Precision  : {metrics["precision"]:.4f}'
            )

            print(
                f'Recall     : {metrics["recall"]:.4f}'
            )

            print(
                f'F1 Score   : {metrics["f1"]:.4f}'
            )

            # =========================================
            # EARLY STOPPING
            # =========================================

            save_path = os.path.join(
                checkpoint_dir,
                "best_model.pth"
            )

            self.early_stopper.step(
                metrics["f1"],
                self.model,
                save_path
            )

            if self.early_stopper.should_stop:

                print(
                    "\n🛑 Early stopping triggered"
                )

                break
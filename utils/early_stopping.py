import os
import torch


class EarlyStopping:

    def __init__(
        self,
        patience=5,
        mode="max"
    ):

        self.patience = patience

        self.mode = mode

        self.best_score = None

        self.counter = 0

        self.should_stop = False

    def step(
        self,
        score,
        model,
        save_path
    ):

        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(model, save_path)

            return

        improved = (
            score > self.best_score
            if self.mode == "max"
            else score < self.best_score
        )

        if improved:

            self.best_score = score

            self.counter = 0

            self.save_checkpoint(model, save_path)

        else:

            self.counter += 1

            print(
                f"No improvement for {self.counter} epoch(s)"
            )

            if self.counter >= self.patience:

                self.should_stop = True

    def save_checkpoint(
        self,
        model,
        save_path
    ):

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        if isinstance(model, torch.nn.DataParallel):

            torch.save(
                model.module.state_dict(),
                save_path
            )

        else:

            torch.save(
                model.state_dict(),
                save_path
            )

        print("✅ Best model saved")
import torch.nn as nn

import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass"
        )

        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass"
        )

    def forward(self, logits, targets):

        dice = self.dice_loss(
            logits,
            targets
        )

        focal = self.focal_loss(
            logits,
            targets
        )

        return dice + focal
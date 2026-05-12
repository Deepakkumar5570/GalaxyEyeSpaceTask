import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from segmentation_models_pytorch.decoders.unet.decoder import (
    UnetDecoder
)


class SiameseUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # =====================================================
        # SHARED ENCODER
        # =====================================================

        self.encoder = smp.encoders.get_encoder(

            name="resnet34",

            in_channels=3,

            depth=5,

            weights="imagenet"

        )

        # =====================================================
        # DECODER
        # =====================================================

        self.decoder = UnetDecoder(

            encoder_channels=[
                c * 2 for c in self.encoder.out_channels
            ],

            decoder_channels=(
                256,
                128,
                64,
                32,
                16
            ),

            n_blocks=5
        )

        # =====================================================
        # SEGMENTATION HEAD
        # =====================================================

        self.segmentation_head = nn.Conv2d(
            16,
            2,
            kernel_size=1
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        pre_image,
        post_image
    ):

        pre_features = self.encoder(pre_image)

        post_features = self.encoder(post_image)

        fused_features = []

        for pre_feat, post_feat in zip(
            pre_features,
            post_features
        ):

            fused = torch.cat(
                [pre_feat, post_feat],
                dim=1
            )

            fused_features.append(fused)

        decoder_output = self.decoder(
            *fused_features
        )

        masks = self.segmentation_head(
            decoder_output
        )

        return masks
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from segmentation_models_pytorch.decoders.unet.decoder import (
    UnetDecoder
)


# =========================================================
# CHANNEL ATTENTION
# =========================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction=16
    ):

        super().__init__()

        hidden_channels = max(
            8,
            in_channels // reduction
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=False
        )

        self.relu = nn.ReLU(inplace=False)

        self.fc2 = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=1,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(
            self.relu(
                self.fc1(
                    self.avg_pool(x)
                )
            )
        )

        max_out = self.fc2(
            self.relu(
                self.fc1(
                    self.max_pool(x)
                )
            )
        )

        out = avg_out + max_out

        return self.sigmoid(out)


# =========================================================
# SPATIAL ATTENTION
# =========================================================

class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        x = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        x = self.conv(x)

        return self.sigmoid(x)


# =========================================================
# CBAM
# =========================================================

class CBAM(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = x * self.channel_attention(x)

        x = x * self.spatial_attention(x)

        return x


# =========================================================
# ATTENTION SIAMESE UNET
# =========================================================

class AttentionSiameseUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # =====================================================
        # ENCODER
        # =====================================================

        self.encoder = smp.encoders.get_encoder(

            name="resnet34",

            in_channels=3,

            depth=5,

            weights="imagenet"

        )

        # =====================================================
        # ATTENTION BLOCKS
        # =====================================================

        self.attentions = nn.ModuleList([

            CBAM(c * 2)

            for c in self.encoder.out_channels

        ])

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
        # HEAD
        # =====================================================

        self.segmentation_head = nn.Sequential(

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=False),

            nn.Conv2d(
                16,
                2,
                kernel_size=1
            )
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

        for idx, (pre_feat, post_feat) in enumerate(

            zip(pre_features, post_features)

        ):

            fused = torch.cat(
                [pre_feat, post_feat],
                dim=1
            )

            fused = self.attentions[idx](fused)

            fused_features.append(fused)

        decoder_output = self.decoder(
            fused_features
        )

        masks = self.segmentation_head(
            decoder_output
        )

        return masks
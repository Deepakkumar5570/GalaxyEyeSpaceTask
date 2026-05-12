import segmentation_models_pytorch as smp


class ResNet34UNet(smp.Unet):

    def __init__(
        self,
        in_channels=6,
        num_classes=2
    ):

        super().__init__(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes
        )
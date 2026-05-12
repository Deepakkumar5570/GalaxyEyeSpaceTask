import segmentation_models_pytorch as smp


class BasicUNet(smp.Unet):

    def __init__(
        self,
        in_channels=6,
        num_classes=2
    ):

        super().__init__(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes
        )
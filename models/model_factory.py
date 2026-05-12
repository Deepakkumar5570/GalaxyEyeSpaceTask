from models.unet.basic_unet import BasicUNet
from models.unet.resnet34_unet import ResNet34UNet
from models.unet.siamese_unet import SiameseUNet
from models.attention.attention_siamese_unet import AttentionSiameseUNet


def build_model(config):

    model_name = config["MODEL"]["NAME"]

    if model_name == "basic_unet":
        return BasicUNet()

    elif model_name == "resnet34_unet":
        return ResNet34UNet()

    elif model_name == "siamese_unet":
        return SiameseUNet()

    elif model_name == "attention_siamese":
        return AttentionSiameseUNet()

    else:
        raise ValueError(f"Unknown model: {model_name}")
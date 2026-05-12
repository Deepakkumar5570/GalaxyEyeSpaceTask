from losses.combined_loss import CombinedLoss


def build_loss(config):

    loss_name = config["LOSS"]["NAME"]

    if loss_name == "dice_focal":
        return CombinedLoss()

    else:
        raise ValueError("Unknown loss")
import torch


def save_checkpoint(model, path):

    if isinstance(model, torch.nn.DataParallel):

        torch.save(
            model.module.state_dict(),
            path
        )

    else:

        torch.save(
            model.state_dict(),
            path
        )


def load_checkpoint(model, path, device):

    state_dict = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    return model
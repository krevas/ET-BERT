import torch


def load_model(model, model_path):
    if hasattr(model, "module"):
        model.module.load_state_dict(
            torch.load(model_path), strict=False
        )
    else:
        model.load_state_dict(torch.load(model_path), strict=False)
    return model

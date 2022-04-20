import torch


def load_model(model, model_path, device=torch.device('cpu')):
    if hasattr(model, "module"):
        model.module.load_state_dict(
            torch.load(model_path, map_location=device), strict=False
        )
    else:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return model

import torch
from models_wrappers.models_wrapper_base import ModelWrapper
from model import CNNDenoiser


class MNISTWrapper(ModelWrapper):
    def __init__(self, model_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = CNNDenoiser()

        pretrained_model = torch.load(model_path, map_location=self.device)['net']
        self.model.load_state_dict(pretrained_model, strict=True)

        self.finalize_model()

    def set_std(self, denoiser_name, **kwargs) -> float:
        return float(denoiser_name.split('_n')[-1])

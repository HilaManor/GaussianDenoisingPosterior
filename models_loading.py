
import os
import sys
import torch
from models_wrappers.models_wrapper_base import ModelWrapper


def load_model(denoiser_name: str,
               device: torch.device,
               model_dir_path: str,
               noise_level: float,
               double_precision: bool = False) -> ModelWrapper:
    model_path = os.path.join(model_dir_path, denoiser_name + '.pth')

    if denoiser_name.startswith('dncnn'):
        sys.path.append("KAIR")
        from models_wrappers.kair_wrappers import DnCNNWrapper

        model = DnCNNWrapper(denoiser_name=denoiser_name,
                             device=device,
                             double_precision=double_precision,
                             model_path=model_path)
    elif 'ircnn' in denoiser_name:
        sys.path.append("KAIR")
        from models_wrappers.kair_wrappers import IRCNNWrapper

        model = IRCNNWrapper(denoiser_name=denoiser_name,
                             device=device,
                             double_precision=double_precision,
                             model_path=model_path,
                             noise_level=noise_level)
    elif 'SwinIR' in denoiser_name:
        sys.path.append("KAIR")
        from models_wrappers.kair_wrappers import SwinIRWrapper

        model = SwinIRWrapper(denoiser_name=denoiser_name,
                              device=device,
                              double_precision=double_precision,
                              model_path=model_path)
    elif 'N2V' in denoiser_name:
        sys.path.append('pn2v')
        from models_wrappers.n2v_wrapper import N2VWrapper

        model = N2VWrapper(device=device,
                           double_precision=double_precision,
                           model_path=model_dir_path)
    elif 'DDPM_FFHQ' in denoiser_name:
        sys.path.append('DDPM_FFHQ')
        from models_wrappers.ddpm_wrapper import DiffWrapper

        from_t = int(denoiser_name.split('_')[-1])
        model = DiffWrapper(from_t=from_t,
                            device=device,
                            double_precision=double_precision,
                            model_path=os.path.join('DDPM_FFHQ', 'ffhq.pt'))
    elif "MNIST" in denoiser_name:
        sys.path.append('MNIST')
        from models_wrappers.mnist_wrapper import MNISTWrapper
        model = MNISTWrapper(device=device,
                             double_precision=double_precision,
                             model_path=model_path)
    else:
        raise NotImplementedError(f"No '{denoiser_name}' denoising model available for loading.")

    return model

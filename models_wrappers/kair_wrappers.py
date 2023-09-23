import torch
from models_wrappers.models_wrapper_base import ModelWrapper
from models.network_swinir import SwinIR
from models.network_dncnn import DnCNN
from models.network_dncnn import IRCNN
import numpy as np


class SwinIRWrapper(ModelWrapper):
    def __init__(self, denoiser_name: str, model_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = SwinIR(upscale=1,
                            in_chans=1 if 'gray' in denoiser_name else 3,
                            img_size=128,
                            window_size=8,
                            img_range=1.,
                            depths=[6, 6, 6, 6, 6, 6],
                            embed_dim=180,
                            num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2,
                            upsampler='',
                            resi_connection='1conv').to(self.device)

        pretrained_model = torch.load(model_path, map_location=self.device)['params']
        self.model.load_state_dict(pretrained_model, strict=True)

        self.finalize_model()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        window_size = 8
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        im = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
        im = torch.cat([im, torch.flip(im, [3])], 3)[:, :, :, :w_old + w_pad]
        out = self.model(im)
        return out.clip(0, 1)[..., :img.shape[-2], :img.shape[-1]]


class DnCNNWrapper(ModelWrapper):
    def __init__(self, denoiser_name: str, model_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if denoiser_name == 'dncnn_15' or denoiser_name == 'dncnn_25' or denoiser_name == 'dncnn_50':
            nb = 17
        elif denoiser_name == 'dncnn_gray_blind' or denoiser_name == 'dncnn_color_blind':
            nb = 20
        else:
            raise NotImplementedError(f"No '{denoiser_name}' denoising model available for loading.")

        self.model = DnCNN(in_nc=3 if 'color' in denoiser_name else 1,
                           out_nc=3 if 'color' in denoiser_name else 1,
                           nc=64, nb=nb, act_mode='R').to(self.device)

        pretrained_model = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(pretrained_model, strict=True)

        self.finalize_model()


class IRCNNWrapper(ModelWrapper):
    def __init__(self, denoiser_name: str, model_path: str, noise_level: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = IRCNN(in_nc=3 if 'color' in denoiser_name else 1,
                           out_nc=3 if 'color' in denoiser_name else 1, nc=64)
        current_idx = min(24, int(np.ceil(noise_level/2)-1))  # current_idx+1 th denoiser

        pretrained_model = torch.load(model_path, map_location=self.device)[str(current_idx)]
        self.model.load_state_dict(pretrained_model, strict=True)

        self.finalize_model()

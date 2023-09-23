import torch
from models_wrappers.models_wrapper_base import ModelWrapper
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion import dist_util
import os
from typing import Optional
import torchvision.transforms as T
from PIL import Image


def convert_module_to_double(layer: torch.nn.Module) -> None:
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        layer.weight.data = layer.weight.data.double()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.double()


class DiffWrapper(ModelWrapper):
    def __init__(self, from_t: int, model_path: str, open_ai_logdir: str = 'DDPM_FFHQ/logdir', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        os.environ['OPENAI_LOGDIR'] = os.path.join(open_ai_logdir)

        self.im_size = 256  # 64
        self.dropout = 0.1  # 0.0
        self.learn_sigma = True  # False
        self.num_channels = 256  # 128
        self.num_head_channels = 64  # -1
        self.resblock_updown = True  # False
        self.attention_resolutions = "32,16,8"  # "16,8"
        self.model, self.diffusion = create_model_and_diffusion(
            image_size=self.im_size,
            class_cond=False,
            learn_sigma=self.learn_sigma,
            num_channels=self.num_channels,
            num_res_blocks=2,
            channel_mult="",
            num_heads=4,
            num_head_channels=self.num_head_channels,
            num_heads_upsample=-1,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=self.resblock_updown,
            use_fp16=False,
            use_new_attention_order=False,
        )

        pretrained_model = dist_util.load_state_dict(model_path, map_location=self.device)
        self.model.load_state_dict(pretrained_model, strict=True)

        self.finalize_model()
        if self.double_precision:
            self.model.input_blocks.apply(convert_module_to_double)
            self.model.middle_block.apply(convert_module_to_double)
            self.model.output_blocks.apply(convert_module_to_double)
            self.model.dtype = torch.double
        self.from_t = from_t
        # self.model = self.model.to(self.device)

    def get_noise(self) -> torch.Tensor:
        return self.diffusion.sqrt_recipm1_alphas_cumprod[self.from_t]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        xt = img * self.diffusion.sqrt_alphas_cumprod[self.from_t]

        t = torch.tensor([self.from_t] * img.shape[0], device=self.device)
        out = self.diffusion.p_sample(
            self.model,
            xt,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
        )
        return out['pred_xstart']

    def set_std(self, noise_std: Optional[float], **kwargs) -> float:
        if noise_std is not None:
            raise ValueError("Sigma is calculated from the DDPM step chosen!")
        return self.get_noise()

    def toim(self, img: torch.Tensor) -> Image.Image:
        if not (img.min() >= 0 and img.max() <= 1):
            img = ((img + 1) / 2).clip(0, 1)
        untensor = T.ToPILImage()
        return untensor(img)

    def save_im(self, img: Image.Image, path: str) -> None:
        img.save(path)

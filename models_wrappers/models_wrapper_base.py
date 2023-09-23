import torch
from typing import Union
from typing import Optional
from torchvision import transforms as T
from PIL import Image


class ModelWrapper(torch.nn.Module):
    def __init__(self, device: torch.device, double_precision: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.double_precision = double_precision
        self.device = device

    def get_noise(self) -> Union[float, torch.Tensor]:
        pass

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

    def finalize_model(self) -> None:
        if self.double_precision:
            self.model = self.model.to(torch.double)
        self.model = self.model.to(self.device)
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False

    def set_std(self, noise_std: Optional[float], **kwargs) -> Optional[float]:
        if noise_std is None:
            raise ValueError("Noise sigma wasn't provided and is needed to add noise to the DB")
        return noise_std

    def toim(self, img: torch.Tensor) -> Image.Image:
        untensor = T.ToPILImage()
        img = img.clip(0, 1)
        return untensor(img)

    def save_im(self, img: Image.Image, path: str) -> None:
        img.save(path)

    @property
    def is_FMD(self):
        return False

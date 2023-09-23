import torch
from models_wrappers.models_wrapper_base import ModelWrapper
from pn2v import prediction
from typing import Optional
from PIL import Image
import numpy as np


class N2VWrapper(ModelWrapper):
    def __init__(self, model_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.load(model_path, map_location=self.device)
        self.finalize_model()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = img * 255.
        res = None

        if len(img.shape) == 2:
            # res = prediction.tiledPredict(img, self.model ,ps=256, overlap=48,
            #    device=self.device, noiseModel=None)
            res = prediction.predict(img.unsqueeze(0), self.model, noiseModel=None, device=self.device,
                                     outScaling=10.0)[0]
        elif len(img.shape) == 3 and (img.shape[0] == 1):
            # res = prediction.tiledPredict(img[0], self.model ,ps=256, overlap=48,
            #    device=self.device, noiseModel=None)
            res = prediction.predict(img, self.model, noiseModel=None, device=self.device, outScaling=10.0)[0]
        elif len(img.shape) == 4 and img.shape[1] == 1:
            outs = []
            for i in range(img.shape[0]):
                # outs.append(prediction.tiledPredict(img[i, 0], self.model ,ps=256, overlap=48,
                # device=self.device, noiseModel=None))
                outs.append(prediction.predict(img[i], self.model, noiseModel=None, device=self.device,
                                               outScaling=10.0)[0])
            res = torch.stack(outs)
        else:
            raise ValueError("N2V dimensions don't fit")
        return res / 255.

    def set_std(self, noise_std: Optional[float], **kwargs) -> Optional[float]:
        if noise_std is not None:
            sure = input("Non-blind noise sigma was provided. "
                         "Are you sure you want to use the provided std? (y/N) ").lower()
            if sure != 'y':
                print("Using estimated sigma")
                return None
        return noise_std

    def toim(self, img: torch.Tensor) -> Image.Image:
        if img.max().item() <= 1.0:
            img = img * 255
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return Image.fromarray(img)

    def save_im(self, img: Image.Image, path: str) -> None:
        img.save(path)

    @property
    def is_FMD(self):
        return True

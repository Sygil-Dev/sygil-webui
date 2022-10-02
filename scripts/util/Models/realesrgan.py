from scripts.util.ModelRepo.model_loader import ModelLoader
from typing import Any, Optional
import os
import sys
import torch


class RealESRGAN(ModelLoader):
    def __init__(self, esrgan_dir: str, model_name: str, half_precision: bool = True, path: str = None, **kwargs):
        super().__init__(**kwargs)
        self._esrgan_dir = esrgan_dir
        self._model_name = model_name
        self._half_precision = half_precision
        self._path = path if path else os.path.join(
            self._esrgan_dir, "experiments/pretrained_models", self._model_name + '.pth')

    def load(self) -> Any:
        """ Overrides ModelLoader.load """
        from basicsr.archs.rrdbnet_arch import RRDBNet
        RealESRGAN_models = {
            'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        }

        sys.path.append(os.path.abspath(self._esrgan_dir))
        from realesrgan import RealESRGANer
        # TODO: this stuffs should be handled by repo
        if self.cpu_only:
            instance = RealESRGANer(
                scale=2, model_path=self._esrgan_dir, model=RealESRGAN_models[self._model_name],
                pre_pad=0, half=False)  # cpu does not support half
            instance.device = torch.device('cpu')
            instance.model.to('cpu')
        else:
            instance = RealESRGANer(
                scale=2, model_path=self._get_model_path(), model=RealESRGAN_models[self._model_name],
                pre_pad=0, half=self._half_precision, device=self.device)
        instance.model.name = self._model_name
        return instance

    def exists(self) -> bool:
        """ Overrides ModelLoader.exists """
        return os.path.isfile(self._get_model_path())

    def _get_model_path(self) -> str:
        return self._path

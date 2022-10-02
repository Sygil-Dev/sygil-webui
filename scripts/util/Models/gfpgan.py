from scripts.util.ModelRepo.model_loader import ModelLoader
from typing import Any, Optional
import os
import sys
import torch

class GFPGAN(ModelLoader):
    def __init__(self, gfpgan_dir: str, model_name: str = "GFPGANv1.3", path: str = None, **kwargs):
        super().__init__(**kwargs)
        self._gfpgan_dir = gfpgan_dir
        self._model_name = model_name
        self._path = path if path else os.path.join(
            self._gfpgan_dir, 'experiments/pretrained_models', self._model_name + '.pth')

    def load(self) -> Any:
        """ Overrides ModelLoader.load """
        sys.path.append(os.path.abspath(self._gfpgan_dir))
        from gfpgan import GFPGANer

        # Workaround: retinaface module holds a 'device' that will keep overwriting GFPGANer's device
        # Just change the module default device
        from facexlib.detection import retinaface
        retinaface.device = self.device
        # End Workaround

        instance = GFPGANer(model_path=self._get_model_path(), upscale=1, arch='clean',
                            channel_multiplier=2, bg_upsampler=None, device=self.device)
        return instance

    def exists(self) -> bool:
        """ Overrides ModelLoader.load """
        return os.path.isfile(self._get_model_path())

    def _get_model_path(self) -> str:
        return self._path

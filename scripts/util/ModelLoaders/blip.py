from scripts.util.ModelRepo.model_loader import ModelLoader
from typing import Any
import os


class BLIP(ModelLoader):
    image_eval_size = 512

    def __init__(self, model_path: str = "models/blip/model__base_caption.pth",
                 config_path: str = "configs/blip/med_config.json", **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._config_path = config_path

    def load(self) -> Any:
        """ Overrides ModelLoader.load """
        #blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
        from ldm.models.blip import blip_decoder
        model = blip_decoder(pretrained=self._get_model_path(),
                             image_size=BLIP.image_eval_size, vit='base', med_config=self._get_config_path())
        model.half()
        return model

    def exists(self) -> bool:
        """ Overrides ModelLoader.exists """
        return os.path.isfile(self._get_model_path()) and os.path.isfile(self._get_config_path())

    def _get_model_path(self) -> str:
        return self._model_path

    def _get_config_path(self) -> str:
        return self._config_path

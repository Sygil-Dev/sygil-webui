from scripts.util.ModelRepo.model_loader import ModelLoader
from typing import Any, Optional
import os
import sys
import torch


class LDSR(ModelLoader):
    def __init__(self, ldsr_dir: str, model_name: str = "model", yaml_name: str = "project", **kwargs):
        super().__init__(**kwargs)
        self._ldsr_dir = ldsr_dir
        self._model_name = model_name
        self._yaml_name = yaml_name
        self._TODO_gpu = 0

    def load(self) -> Any:
        """ Overrides ModelLoader.load """
        sys.path.append(os.path.abspath(self._ldsr_dir))
        from LDSR import LDSR
        return LDSR(self._get_model_path(), self._get_yaml_path())

    def exists(self) -> bool:
        """ Overrides ModelLoader.load """
        return os.path.isfile(self._get_model_path()) and os.path.isfile(self._get_yaml_path())

    def _get_model_path(self) -> str:
        return os.path.join(self._ldsr_dir, 'experiments/pretrained_models', self._model_name + '.ckpt')

    def _get_yaml_path(self) -> str:
        return os.path.join(self._ldsr_dir, 'experiments/pretrained_models', self._yaml_name + '.yaml')

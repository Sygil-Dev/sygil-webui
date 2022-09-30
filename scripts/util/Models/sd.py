from scripts.util.ModelRepo.model_loader import ModelLoader
from typing import Any, Optional
import os
import sys
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from ldm.util import instantiate_from_config
from functools import lru_cache


class SD(ModelLoader):
    def __init__(self, checkpoint: str, config_yaml: str, half_precision: bool = True, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._checkpoint: str = checkpoint
        self._config_yaml: str = config_yaml
        self._half_precision = half_precision
        self._verbose: bool = verbose
        self._TODO_gpu = 0

    def load(self) -> Any:
        """ Overrides ModelLoader.load """
        config = OmegaConf.load(self._get_yaml_path())
        model = self._model_from_config(model_path=self._get_model_path(), config=config, verbose=self._verbose)
        if self._half_precision:
            model = model.half()
        model = model.eval()
        return model

    def exists(self) -> bool:
        """ Overrides ModelLoader.exists """
        return os.path.isfile(self._get_model_path()) and os.path.isfile(self._get_yaml_path())

    def _model_from_config(self, model_path: str, config: DictConfig, verbose: bool = False) -> Any:
        """Create the model from a given path and omegaconf

        Args:
            model_path (str): path to model checkpoint
            config (DictConfig): configuration to use
            verbose (bool, optional): verbose logging

        Returns:
            Any: model instance
        """
        sd = SD._load_sd_model(model_path)
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        return model

    @classmethod
    @lru_cache(maxsize=1)
    def _load_sd_model(cls, model_path: str):
        """ Cached wrapper for torch.load to share the same model """
        pl_sd = torch.load(model_path, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        return sd

    def _get_model_path(self) -> str:
        return self._checkpoint

    def _get_yaml_path(self) -> str:
        return self._config_yaml

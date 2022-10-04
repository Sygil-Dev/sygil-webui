from . sd import SD
from typing import Any, Optional
import os
import sys
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from enum import Enum, auto
from omegaconf.dictconfig import DictConfig


class SD_Optimized(SD):
    class Stage(Enum):
        UNET = auto(),
        COND_STAGE = auto(),
        FIRST_STAGE = auto()

    def __init__(self, stage: Stage, **kwargs):
        super().__init__(**kwargs)
        self._stage = stage
        self._config_yaml = "optimizedSD/v1-inference.yaml"

    def _model_from_config(self, model_path: str, config: DictConfig, verbose: bool = False) -> Any:
        """ Overrides SD._model_from_config
        Returns Optimized versions of the SD model """
        sd = SD._load_sd_model(model_path)
        li, lo = [], []
        for key, v_ in sd.items():
            sp = key.split('.')
            if (sp[0]) == 'model':
                if ('input_blocks' in sp):
                    li.append(key)
                elif ('middle_block' in sp):
                    li.append(key)
                elif ('time_embed' in sp):
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd['model1.' + key[6:]] = sd.pop(key)
        for key in lo:
            sd['model2.' + key[6:]] = sd.pop(key)

        model = None
        if self._stage == SD_Optimized.Stage.UNET:
            model = instantiate_from_config(config.modelUNet)
            _, _ = model.load_state_dict(sd, strict=False)
        elif self._stage == SD_Optimized.Stage.COND_STAGE:
            model = instantiate_from_config(config.modelCondStage)
            model.cond_stage_model.device = torch.device('cuda')  # FIXME: necessary on turbo since repo won't recurse?
            _, _ = model.load_state_dict(sd, strict=False)
        elif self._stage == SD_Optimized.Stage.FIRST_STAGE:
            model = instantiate_from_config(config.modelFirstStage)
            _, _ = model.load_state_dict(sd, strict=False)
        else:
            raise Exception(f"Unknown stage {self._stage}")
        if self._half_precision:
            model = model.half()
        return model

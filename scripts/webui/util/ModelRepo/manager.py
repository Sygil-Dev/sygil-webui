
''' Attempts to help manage torch model GPU memory usage '''
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Generator
from threading import Thread, Event
import sys
import traceback
import torch
from .model import Model, ModelHndl


@dataclass
class ModelInfo:
    model: Model
    check_func: Callable
    ref_cnt: int = 0


class Manager:
    def __init__(self, device: torch.device):
        """Construct a new ModelRepo Manager

        Args:
            device (torch.device): device to use
        """
        self._device = device
        self._model_infos: Dict[str, ModelInfo] = {}

    def register_model(
            self, name: str, load_func: Callable, exists_func: Callable, preload: bool = False,
            load_kwargs: Dict[str, Any] = None):
        """Registers a new model with the model manager

        Args:
            name (str): name to register the model as
            load_func (Callable): the 'load' function that initializes and returns
                the registered model.
            exists_func (Callable): a function to quickly check for model availability
                without fully loading the model.
                This function will be called with the same parameters that load_func would
                receive. It should return True if the model exists
            load_kwargs (Dict): keyword arguments to pass to the load function
            preload (bool): immediately begin loading the model from disk
        """
        if name in self._model_infos:
            raise KeyError(f"Model name {name} is already registered")
        model = Model(name=name, load_func=load_func, exists_func=exists_func, load_kwargs=load_kwargs or {})
        self._model_infos[name] = ModelInfo(model=model, check_func=exists_func)

        if preload:
            model.load()

    def is_loadable(self, name: str) -> bool:
        """Returns quick check if model is registered and appears loadable

        Args:
            name (str): registered name of the model

        Returns:
            bool: True if the model's check_func succeeds
        """
        model_info = self._model_infos.get(name, None)
        if model_info:
            try:
                return model_info.model.exists()
            except Exception as e:
                print(f"Checking [{name}] failed with [{e}]:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        return False

    @contextmanager
    def model_context(self, *args, **kwargs) -> Generator[ModelHndl, None, None]:
        """Access a model using a 'with' context.
           Use arguments as if calling get_model
        """
        hndl = None
        try:
            hndl = self.get_model(*args, **kwargs)
        finally:
            yield hndl
            if hndl:
                del hndl


    def get_model(self, name: str) -> Optional[ModelHndl]:
        """Returns a handle to the requested model if it is registered, else None
           The handle can be used as if it was the model itself. When no longer
           needed, the model should be deleted with 'del'

        Args:
            name (str): model to load
            on_device (bool): if true require running on device
        """
        model_info = self._model_infos.get(name, None)
        if not model_info:
            return None

        model_info.ref_cnt += 1

        # Aggressively move models off device
        for key,info in self._model_infos.items():
            if key != name:
                info.model.move_model(to_device=False)
        torch_gc()

        # Ensure the model is properly loaded
        model_info.model.load()
        return model_info.model.get_handle(
            del_callback=partial(self._handle_hndl_closed, model_info=model_info)
        )

    def _handle_hndl_closed(self, model_info: ModelInfo):
        """Callback when a handle is closed. Hook to perform bookkeeping

        Args:
            model_info (ModelInfo): the handle being closed
        """
        assert model_info.ref_cnt > 0
        model_info.ref_cnt -= 1
        if model_info.ref_cnt == 0:
            print(f"No more refs for {model_info.model._name}")

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
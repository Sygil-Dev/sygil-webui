""" Wrapper for models which gives ModelRepo hooks to manage them """
from __future__ import annotations

import inspect
import logging
import sys
import time
import uuid
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional

import torch

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("ModelRepo.Model")


class Model(object):
    def __init__(
            self, name: str, load_func: Callable, exists_func: Callable, load_kwargs: Optional[Dict] = None):
        """Initialize a new Model

        Args:
            name (str): label to identify the model
            load_func (Callable): function to call, with load_kwargs parameters, to instantiate the model
            exists_func (Callable): function to call, with load_kwargs parameters, to quickly check if the model exists
            load_kwargs (Optional[Dict], optional): keyword parameters to pass to the load/exists functions
        """
        self._name: str = name

        # Underlying model and related functions
        self._instance: Any = None
        self._load_func: Callable = load_func
        self._exists_func: Callable = exists_func
        self._load_kwargs: Dict = load_kwargs
        self._child_models: Dict[str, torch.nn.Module] = {}

        self._loaded: Event = Event()
        self._on_device: bool = False

    def exists(self) -> bool:
        """Check if the model appears to exist without actually loading the model"""
        try:
            exists = self._exists_func(**self._load_kwargs)
        except Exception as e:
            exists = False
        return exists

    def is_loaded(self, block=False) -> bool:
        """Returns if the model has been loaded from disk.
        If block is true then waits until the model is loaded"""
        return self._loaded.wait(timeout=None if block else 0)

    def load(self):
        """Load the model from disk """
        logger.info(f"Loading model {self._name}")
        try:
            start = time.time()
            if not self._instance:
                self._instance = self._load_func(**self._load_kwargs)

                # Attributes to skip when evaluating children
                # eg, model_size is a very expensive property to even getattr
                BLACKLIST: set(str) = set(["model_size"])

                # Find any children models
                for key in dir(self._instance):
                    if key in BLACKLIST:
                        continue
                    attr = getattr(self._instance, key)
                    if isinstance(attr, torch.nn.Module):
                        self._child_models[key] = attr

                if hasattr(self._instance, 'eval'):
                    self._instance.eval()
                else:
                    logger.warning(f"no eval {self._name}")

                self._loaded.set()
            logger.debug(f"Model {self._name} loaded! Took {time.time()-start:.3f} seconds")
        except Exception as e:
            logger.error(f"Failed to load model {self._name}: {e}")

    def move_model(self, to_device: bool = False):
        """Moves the model between CPU and device

        Args:
            to_device (bool, optional): if true then move the model to the device, otherwise CPU
        """
        if to_device == self._on_device:
            return
        logger.debug(f"Moving {self._name} to (device: {to_device})")
        start_time = time.time()
        # TODO: Move child models as well?
        if to_device:
            for model in [self._instance]:
                if hasattr(model, 'to'):
                    model.to('cuda')
            self._on_device = True
        else:
            for model in [self._instance]:
                if hasattr(model, 'to'):
                    model.to('cpu')
            self._on_device = False
        logger.debug(
            f"Done moving {self._name} toDevice {to_device}. Elapsed time: {time.time()-start_time:.3f} seconds")

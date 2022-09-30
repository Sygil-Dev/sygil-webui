
''' Attempts to help manage torch model GPU memory usage '''
from __future__ import annotations

import sys
import time
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Generator, List, Optional

import torch
from readerwriterlock import rwlock

from .model import Model
from .model_loader import ModelLoader
from .schedulers import Scheduler, OneAtATimeScheduler

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("ModelRepo.Manager")


@dataclass
class ModelInfo:
    model: Model
    check_func: Callable
    ref_cnt: int = 0
    loading_lock: Lock = field(default_factory=Lock)

    depth_remaining: int = 0
    scheduler_token: int = None


class Manager:
    NUM_WORKERS: int = 5

    def __init__(self, device: torch.device = torch.device("cuda"), scheduler: Optional[Scheduler] = None):
        """Construct a new ModelRepo Manager

        Args:
            device (torch.device): device to use
            scheduler (Optional[Scheduler]): Scheduler instance to use to  manage memory policy
        """
        if scheduler is None:
            scheduler = OneAtATimeScheduler()
        self._device = device
        self._model_infos: Dict[str, ModelInfo] = {}
        self._model_info_lock: rwlock.RWLockWrite = rwlock.RWLockWrite()
        self._executor = ThreadPoolExecutor(Manager.NUM_WORKERS)
        self._scheduler: Scheduler = scheduler

    def register_model_loader(self, name: str, loader: ModelLoader, **reg_kwargs):
        """Register a model via a ModelLoader
        Additional keyword arguments are passed on to register_model

        Args:
            name (str): name to register model as
            loader (ModelLoader): _description_

        """
        self.register_model(name=name, load_func=loader.load, exists_func=loader.exists, **reg_kwargs)

    def register_model(
            self, name: str, load_func: Callable, exists_func: Callable, preload: bool = False, max_depth: int = 1,
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
        scheduler_token = self._scheduler.register(model)
        model_info = ModelInfo(model=model, check_func=exists_func,
                               depth_remaining=max_depth, scheduler_token=scheduler_token)
        with self._model_info_lock.gen_wlock():
            self._model_infos[name] = model_info
        if preload:
            self._load_model(model_info)
        logger.info(f"Model {name} has been registered")

    def is_loadable(self, name: str) -> bool:
        """Returns quick check if model is registered and appears loadable"""
        model_info = self._model_infos.get(name, None)
        if model_info:
            try:
                return model_info.model.exists()
            except Exception as e:
                logger.error(f"Checking [{name}] failed with [{e}]:")
                logger.error(traceback.format_exc())
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

        # Ensure the model is properly loaded
        self._load_model(model_info)

        return ModelHndl(
            model=model_info.model,
            del_callback=partial(self._handle_modelhndl_closed, model_info=model_info),
            call_filter=partial(self._call_wrapper, model_info)
        )

    def _handle_modelhndl_closed(self, model_info: ModelInfo):
        """Callback when a handle is closed. Hook to perform bookkeeping

        Args:
            model_info (ModelInfo): the handle being closed
        """
        assert model_info.ref_cnt > 0
        model_info.ref_cnt -= 1
        if model_info.ref_cnt == 0:
            logging.debug(f"No more refs for {model_info.model._name}")

    def _load_model(self, model_info: ModelInfo, block=False) -> None:
        """Starts loading the model from disk into RAM

        Args:
            model_info (ModelInfo): model_info to ensure is loaded from disk
            block (bool, optional): if true, blocks until the model is loaded
        """
        # TODO: Replace this with a rwlock?
        # TODO: The scheduler could do this?
        if not model_info.model.is_loaded():
            if model_info.loading_lock.acquire(blocking=block):
                if not model_info.model.is_loaded():
                    def task():
                        try:
                            start = time.time()
                            model_info.model.load()

                            self._register_inner_models(model_info)
                        except Exception as e:
                            logging.error(f"Exception in task: {e}")
                            raise
                        finally:
                            model_info.loading_lock.release()
                    ret = self._executor.submit(task)
                    if block:
                        ret.result()

    def _register_inner_models(self, model_info: ModelInfo) -> None:
        """Recursively searches model instances for child models, registering them
           for management if found.

           Recurses until model_info.depth_remaining is 0.

        Args:
            model_info (ModelInfo): registered model to scan for child models
        """
        if model_info.depth_remaining == 0:
            return

        for name, child_model in model_info.model._child_models.items():
            # Register the child model
            model_name = f"{model_info.model._name}##{name}"
            self.register_model(name=model_name,
                                load_func=lambda x=child_model: x,
                                exists_func=lambda: True,
                                max_depth=model_info.depth_remaining - 1
                                )
            # Now get a wrapped handle to that registered model
            new_hndl = self.get_model(name=model_name)
            # And replace the child with the wrapped instance
            delattr(model_info.model._instance, name)
            setattr(model_info.model._instance, name, new_hndl)

    # Whitelist of function calls which can be made without requiring the model be moved to the device
    OnCpuWhitelist = set()  # ['embedding_manager', 'get_learned_conditioning', '__name__', '__qualname__'])

    def _call_wrapper(self, model_info: ModelInfo, attr: str) -> Any:
        """Handles calls to attributes on ModelHndl by ensuring model
        is loaded from disk then running the call thorugh the scheduler

        Args:
            model_info (ModelInfo): object that the attr is being called on
            attr (str): attribute being referenced on the object

        Returns:
            Any: return value from the wrapped function
        """
        # Ensure the model is loaded from disk
        if not model_info.model.is_loaded():
            logger.debug(f"{model_info.model._name} call to {attr} blocked to load model from disk")
            self._load_model(model_info, block=True)

        # Then get the attribute from the underlying instance.
        # If it is a function then call it via the scheduler, otherwise just return it
        ret = getattr(model_info.model._instance, attr)
        if not isinstance(ret, ModelHndl) and callable(ret):

            wrapped_func = ret

            def scheduler_wrapped_func(*args, **kwargs):
                initial_retry_cnt = 10
                retries_left = initial_retry_cnt
                while True:
                    with self._scheduler.on_device_context(model_info.scheduler_token):
                        try:
                            wrapped_ret = wrapped_func(*args, **kwargs)
                        except RuntimeError as e:
                            # Attempt to retry after an out of memory error
                            if retries_left == initial_retry_cnt:
                                logger.error(
                                    f"Exception running wrapped function {attr} on {model_info.model._instance}")
                                logger.error(traceback.format_exc())
                            if retries_left > 0:
                                logger.warning(
                                    f"Retrying function call [{attr} on {model_info.model._instance}]. This may produce unexpected results. Retries left: {retries_left}")
                                retries_left -= 1
                                continue
                            raise

                    return wrapped_ret
            ret = scheduler_wrapped_func

        if isinstance(ret, torch.Tensor) and ret.device.type == 'cpu':
            logger.warning(f"Wrapper moved tensor returned by {attr} on {model_info.model._instance} to device")
            ret = ret.cuda()  # Don't go around returning cpu tensors
        return ret


class ModelHndl(object):
    """ Handle to a Model which can be returned to clients"""

    def __init__(self, model: Model, del_callback: Optional[Callable] = None, call_filter: Optional[Callable] = None):
        self._model = model
        self._del_callback = del_callback
        if call_filter is None:
            def call_filter(attr):
                return getattr(self._model._instance, attr)
        self._call_filter = call_filter

    def __del__(self):
        if callable(self._del_callback):
            self._del_callback()

    def __call__(self, *args, **kwargs):
        return self._call_filter('__call__')(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self._call_filter('__iter__')(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._call_filter('__getitem__')(*args, **kwargs)

    def __getattr__(self, attr):
        # Check if ModelHndl can handle the call
        if attr in self.__dict__:
            return getattr(self, attr)

        # Otherwise forward the call
        ret = self._call_filter(attr)
        return ret


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

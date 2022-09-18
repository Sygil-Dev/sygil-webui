""" Wrapper for models which gives ModelRepo hooks to manage them """
from __future__ import annotations
from threading import Thread, Event
from typing import Any, Callable, Dict, Optional
import time
import uuid


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

        self._loading: Event = Event()
        self._loaded: Event = Event()
        self._load_worker: Thread = None
        self._on_device: bool = False

    def exists(self) -> bool:
        """Check if the model appears to exist without actually loading the model"""
        try:
            exists = self._exists_func(**self._load_kwargs)
        except Exception as e:
            exists = False
        return exists

    def get_handle(self, del_callback: Optional[Callable] = None) -> ModelHndl:
        """Gets a handle to the model which will call del_callback when freed

        Args:
            del_callback (Optional[Callable]): Callable to call when handle is released

        Returns:
            ModelHndl: ModelHndl, which forwards calls to the underlying model
        """
        return ModelHndl(self, del_callback=del_callback)

    def is_loaded(self) -> bool:
        """Returns if the model has been loaded from disk"""
        return self._loaded.is_set() and self._instance is not None

    def load(self, block: bool = False):
        """Begins loading the model from disk on a background thread

        Args:
            block (bool, optional): if True, block until the model has finished loading
        """
        if not self.is_loaded():
            if not self._loading.is_set():
                self._loaded.clear()
                self._loading.set()
                self._load_worker = Thread(name=f"Load {self._name}", target=self._load_worker_func)
                self._load_worker.start()

        if block:
            self._loaded.wait()

    def _load_worker_func(self):
        """Background worker thread function to load model"""
        print(f"Background thread loading model {self._name}")
        try:
            start = time.time()
            if not self._instance:
                self._instance = self._load_func(**self._load_kwargs)
                self._instance.eval()
            print(f"Model {self._name} loaded! Took {time.time()-start:.3f} seconds")
        except Exception as e:
            print(f"Failed to load model {self._name}: {e}")
        finally:
            self._loaded.set()
            self._loading.clear()
            self._load_worker = None

    def move_model(self, to_device: bool = False):
        """Moves the model between CPU and device

        Args:
            to_device (bool, optional): if true then move the model to the device, otherwise CPU
        """
        if to_device == self._on_device:
            return
        print(f"Moving {self._name} to (device: {to_device})")
        start_time = time.time()
        if to_device:
            self._instance.cuda()
            self._on_device = True
        else:
            self._instance.cpu()
            self._on_device = False
        print(f"Done. Elapsed time: {time.time()-start_time:.3f} seconds")


class ModelHndl(object):
    """ Handle to a Model which can be returned to clients"""

    # Whitelist of calls that do not require moving the model to the GPU
    OnCpuWhitelist = set(['embedding_manager'])

    def __init__(self, model: Model, del_callback: Optional[Callable] = None):
        self._model = model
        self._del_callback = del_callback

    def __del__(self):
        if callable(self._del_callback):
            self._del_callback()

    def __getattr__(self, attr):
        # Check if ModelHndl can handle the call
        if attr in self.__dict__:
            return getattr(self, attr)

        # Otherwise it is for the model. Make sure it is loaded from disk
        if not self._model.is_loaded():
            print(f"{self._model._name} Blocking on call to {attr} to load model")
            self._model.load(block=True)

        print(f"{self._model._name} forwarding call to for {attr}]")
        if attr not in ModelHndl.OnCpuWhitelist:
            self._model.move_model(to_device=True)

        ret = getattr(self._model._instance, attr)
        print(f"RetType: {type(ret)}")
        return ret

    def _on_device_wrapper(self, func: Callable, *args, **kwargs):
        ret = func(*args, **kwargs)
        return ret

""" Wraps an already-loaded model in a ModelLoader """
from scripts.util.ModelRepo.model_loader import ModelLoader
from typing import Any


class Wrapper(ModelLoader):
    def __init__(self, model: Any, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    def load(self) -> Any:
        """ Overrides ModelLoader.load """
        return self._model

    def exists(self) -> bool:
        """ Overrides ModelLoader.exists """
        return self._model is not None

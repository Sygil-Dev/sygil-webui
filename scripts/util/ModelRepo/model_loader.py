from typing import Any
import torch


class ModelLoader:
    def __init__(self, device: torch.device = torch.device("cuda"), max_depth: int = 0):
        """Initialize a ModelLoader for use with ModelRepo

        Args:
            cpu_only (bool, optional): Run this model only on the CPU
            max_depth (int, optional): maximum child depth for ModelRepo to attempt hooking the model
        """
        self._device = device
        self._max_depth = max_depth

    def load(self) -> Any:
        """ Loads the model from disk
        Returns the loaded instance """
        raise NotImplementedError("Load must be implemented in a derived class")

    def exists(self) -> bool:
        """ Quick check if the model appears to exist, without fully loading from disk
        Returns True if the model appears to exists """
        raise NotImplementedError("Load must be implemented in a derived class")

    @property
    def is_cpu(self) -> bool:
        return self.device.type == 'cpu'

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def max_depth(self) -> int:
        return self._max_depth

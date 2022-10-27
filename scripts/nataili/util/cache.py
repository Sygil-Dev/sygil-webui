import gc

import torch
import threading
import pynvml
import time

with torch.no_grad():
    def torch_gc():
        for _ in range(2):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

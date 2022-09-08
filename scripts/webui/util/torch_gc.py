from util.imports import *

def torch_gc():
	torch.cuda.empty_cache()
	torch.cuda.ipc_collect()
from util.imports import *

def load_GFPGAN():
	model_name = 'GFPGANv1.3'
	model_path = os.path.join(defaults.general.GFPGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
	if not os.path.isfile(model_path):
		raise Exception("GFPGAN model not found at path "+model_path)

	sys.path.append(os.path.abspath(defaults.general.GFPGAN_dir))
	from gfpgan import GFPGANer

	if defaults.general.gfpgan_cpu or defaults.general.extra_models_cpu:
		instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device('cpu'))
	elif defaults.general.extra_models_gpu:
		instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device(f'cuda:{defaults.general.gfpgan_gpu}'))
	else:
		instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device(f'cuda:{defaults.general.gpu}'))
	return instance
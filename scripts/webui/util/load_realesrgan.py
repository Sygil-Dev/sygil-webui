from util.imports import *

def load_RealESRGAN(model_name: str):
	from basicsr.archs.rrdbnet_arch import RRDBNet
	RealESRGAN_models = {
                'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        }

	model_path = os.path.join(defaults.general.RealESRGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
	if not os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{model_name}.pth")):
		raise Exception(model_name+".pth not found at path "+model_path)

	sys.path.append(os.path.abspath(defaults.general.RealESRGAN_dir))
	from realesrgan import RealESRGANer

	if defaults.general.esrgan_cpu or defaults.general.extra_models_cpu:
		instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=False) # cpu does not support half
		instance.device = torch.device('cpu')
		instance.model.to('cpu')
	elif defaults.general.extra_models_gpu:
		instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not defaults.general.no_half, device=torch.device(f'cuda:{defaults.general.esrgan_gpu}'))
	else:
		instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not defaults.general.no_half, device=torch.device(f'cuda:{defaults.general.gpu}'))
	instance.model.name = model_name

	return instance
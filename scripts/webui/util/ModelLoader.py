from util.imports import *
from util.load_from_config import *
from util.torch_gc import *
from util.load_gfpgan import *
from util.load_realesrgan import *

def ModelLoader(models,load=False,unload=False,imgproc_realesrgan_model_name='RealESRGAN_x4plus'):
	#get global variables
	global_vars = globals()
	#check if m is in globals
	if unload:
		for m in models:
			if m in global_vars:
				#if it is, delete it
				del global_vars[m]
				if defaults.general.optimized:
					if m == 'model':
						del global_vars[m+'FS']
						del global_vars[m+'CS']
				if m =='model':
					m='Stable Diffusion'
				print('Unloaded ' + m)
	if load:
		for m in models:
			if m not in global_vars or m in global_vars and type(global_vars[m]) == bool:
				#if it isn't, load it
				if m == 'GFPGAN':
					global_vars[m] = load_GFPGAN()
				elif m == 'model':
					sdLoader = load_sd_from_config()
					global_vars[m] = sdLoader[0]
					if defaults.general.optimized:
						global_vars[m+'CS'] = sdLoader[1]
						global_vars[m+'FS'] = sdLoader[2]
				elif m == 'RealESRGAN':
					global_vars[m] = load_RealESRGAN(imgproc_realesrgan_model_name)
				elif m == 'LDSR':
					global_vars[m] = load_LDSR()
				if m =='model':
					m='Stable Diffusion'
				print('Loaded ' + m)
	torch_gc()
import warnings
import streamlit as st
from streamlit import StopException, StreamlitAPIException

import base64, cv2
import argparse, os, sys, glob, re, random, datetime
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
import requests
from scipy import integrate
import torch
from torchdiffeq import odeint
from tqdm.auto import trange, tqdm
import k_diffusion as K
import math
import mimetypes
import numpy as np
import pynvml
import threading, asyncio
import time
import torch
from torch import autocast
from torchvision import transforms
import torch.nn as nn
import yaml
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from io import BytesIO
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
     extract_into_tensor
from retry import retry

# we use python-slugify to make the filenames safe for windows and linux, its better than doing it manually
# install it with 'pip install python-slugify'
from slugify import slugify

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass

# remove some annoying deprecation warnings that show every now and then.
warnings.filterwarnings("ignore", category=DeprecationWarning)     

defaults = OmegaConf.load("configs/webui/webui_streamlit.yaml")

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

# should and will be moved to a settings menu in the UI at some point
grid_format = [s.lower() for s in defaults.general.grid_format.split(':')]
grid_lossless = False
grid_quality = 100
if grid_format[0] == 'png':
	grid_ext = 'png'
	grid_format = 'png'
elif grid_format[0] in ['jpg', 'jpeg']:
	grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
	grid_ext = 'jpg'
	grid_format = 'jpeg'
elif grid_format[0] == 'webp':
	grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
	grid_ext = 'webp'
	grid_format = 'webp'
	if grid_quality < 0: # e.g. webp:-100 for lossless mode
		grid_lossless = True
		grid_quality = abs(grid_quality)

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(defaults.general.gpu)

@retry(tries=5)
def load_models(continue_prev_run = False, use_GFPGAN=False, use_RealESRGAN=False, RealESRGAN_model="RealESRGAN_x4plus"):
	"""Load the different models. We also reuse the models that are already in memory to speed things up instead of loading them again. """

	print ("Loading models.")

	# Generate random run ID
	# Used to link runs linked w/ continue_prev_run which is not yet implemented
	# Use URL and filesystem safe version just in case.
	st.session_state["run_id"] = base64.urlsafe_b64encode(
                os.urandom(6)
            ).decode("ascii")

	# check what models we want to use and if the they are already loaded.

	if use_GFPGAN:
		if "GFPGAN" in st.session_state:
			print("GFPGAN already loaded")
		else:
			# Load GFPGAN
			if os.path.exists(defaults.general.GFPGAN_dir):
				try:
					st.session_state["GFPGAN"] = load_GFPGAN()
					print("Loaded GFPGAN")
				except Exception:
					import traceback
					print("Error loading GFPGAN:", file=sys.stderr)
					print(traceback.format_exc(), file=sys.stderr)          
	else:
		if "GFPGAN" in st.session_state:
			del st.session_state["GFPGAN"]        

	if use_RealESRGAN:
		if "RealESRGAN" in st.session_state and st.session_state["RealESRGAN"].model.name == RealESRGAN_model:
			print("RealESRGAN already loaded")
		else:
			#Load RealESRGAN 
			try:
				# We first remove the variable in case it has something there,
				# some errors can load the model incorrectly and leave things in memory.
				del st.session_state["RealESRGAN"]
			except KeyError:
				pass

			if os.path.exists(defaults.general.RealESRGAN_dir):
				# st.session_state is used for keeping the models in memory across multiple pages or runs.
				st.session_state["RealESRGAN"] = load_RealESRGAN(RealESRGAN_model)
				print("Loaded RealESRGAN with model "+ st.session_state["RealESRGAN"].model.name)

	else:
		if "RealESRGAN" in st.session_state:
			del st.session_state["RealESRGAN"]        


	if "model" in st.session_state:
		print("Model already loaded")
	else:
		config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
		model = load_model_from_config(config, defaults.general.ckpt)

		st.session_state["device"] = torch.device(f"cuda:{defaults.general.gpu}") if torch.cuda.is_available() else torch.device("cpu")
		st.session_state["model"] = (model if defaults.general.no_half else model.half()).to(st.session_state["device"] )    

		print("Model loaded.")


def load_model_from_config(config, ckpt, verbose=False):

	print(f"Loading model from {ckpt}")

	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def load_sd_from_config(ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	return sd
#
@retry(tries=5)
def generation_callback(img, i=0):

	try:
		if i == 0:	
			if img['i']: i = img['i']
	except TypeError:
		pass


	if i % int(defaults.general.update_preview_frequency) == 0 and defaults.general.update_preview:
		#print (img)
		#print (type(img))
		# The following lines will convert the tensor we got on img to an actual image we can render on the UI.
		# It can probably be done in a better way for someone who knows what they're doing. I don't.		
		#print (img,isinstance(img, torch.Tensor))
		if isinstance(img, torch.Tensor):
			x_samples_ddim = (st.session_state["model"] if not defaults.general.optimized else modelFS).decode_first_stage(img)          
		else:
			# When using the k Diffusion samplers they return a dict instead of a tensor that look like this:
			# {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised}			
			x_samples_ddim = (st.session_state["model"] if not defaults.general.optimized else modelFS).decode_first_stage(img["denoised"])

		x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)  

		pil_image = transforms.ToPILImage()(x_samples_ddim.squeeze_(0)) 			

		# update image on the UI so we can see the progress
		st.session_state["preview_image"].image(pil_image) 	

	# Show a progress bar so we can keep track of the progress even when the image progress is not been shown,
	# Dont worry, it doesnt affect the performance.	
	if st.session_state["generation_mode"] == "txt2img":
		percent = int(100 * float(i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps)/float(st.session_state.sampling_steps))
		st.session_state["progress_bar_text"].text(
                        f"Running step: {i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps} {percent if percent < 100 else 100}%")
	else:
		round_sampling_steps = round(st.session_state.sampling_steps * st.session_state["denoising_strength"])
		percent = int(100 * float(i+1 if i+1 < round_sampling_steps else round_sampling_steps)/float(round_sampling_steps))
		st.session_state["progress_bar_text"].text(
                        f"""Running step: {i+1 if i+1 < round_sampling_steps else round_sampling_steps}/{round_sampling_steps} {percent if percent < 100 else 100}%""")

	st.session_state["progress_bar"].progress(percent if percent < 100 else 100)



class MemUsageMonitor(threading.Thread):
	stop_flag = False
	max_usage = 0
	total = -1

	def __init__(self, name):
		threading.Thread.__init__(self)
		self.name = name

	def run(self):
		try:
			pynvml.nvmlInit()
		except:
			print(f"[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n")
			return
		print(f"[{self.name}] Recording max memory usage...\n")
		handle = pynvml.nvmlDeviceGetHandleByIndex(defaults.general.gpu)
		self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
		while not self.stop_flag:
			m = pynvml.nvmlDeviceGetMemoryInfo(handle)
			self.max_usage = max(self.max_usage, m.used)
			# print(self.max_usage)
			time.sleep(0.1)
		print(f"[{self.name}] Stopped recording.\n")
		pynvml.nvmlShutdown()

	def read(self):
		return self.max_usage, self.total

	def stop(self):
		self.stop_flag = True

	def read_and_stop(self):
		self.stop_flag = True
		return self.max_usage, self.total

class CFGMaskedDenoiser(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.inner_model = model

	def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
		x_in = x
		x_in = torch.cat([x_in] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_in = torch.cat([uncond, cond])
		uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
		denoised = uncond + (cond - uncond) * cond_scale

		if mask is not None:
			assert x0 is not None
			img_orig = x0
			mask_inv = 1. - mask
			denoised = (img_orig * mask_inv) + (mask * denoised)

		return denoised

class CFGDenoiser(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.inner_model = model

	def forward(self, x, sigma, uncond, cond, cond_scale):
		x_in = torch.cat([x] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_in = torch.cat([uncond, cond])
		uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
		return uncond + (cond - uncond) * cond_scale
def append_zero(x):
	return torch.cat([x, x.new_zeros([1])])
def append_dims(x, target_dims):
	"""Appends dimensions to the end of a tensor until it has target_dims dimensions."""
	dims_to_append = target_dims - x.ndim
	if dims_to_append < 0:
		raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
	return x[(...,) + (None,) * dims_to_append]
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
	"""Constructs the noise schedule of Karras et al. (2022)."""
	ramp = torch.linspace(0, 1, n)
	min_inv_rho = sigma_min ** (1 / rho)
	max_inv_rho = sigma_max ** (1 / rho)
	sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
	return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
	"""Constructs an exponential noise schedule."""
	sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
	return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
	"""Constructs a continuous VP noise schedule."""
	t = torch.linspace(1, eps_s, n, device=device)
	sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
	return append_zero(sigmas)


def to_d(x, sigma, denoised):
	"""Converts a denoiser output to a Karras ODE derivative."""
	return (x - denoised) / append_dims(sigma, x.ndim)
def linear_multistep_coeff(order, t, i, j):
	if order - 1 > i:
		raise ValueError(f'Order {order} too high for step {i}')
	def fn(tau):
		prod = 1.
		for k in range(order):
			if j == k:
				continue
			prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
		return prod
	return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

class KDiffusionSampler:
	def __init__(self, m, sampler):
		self.model = m
		self.model_wrap = K.external.CompVisDenoiser(m)
		self.schedule = sampler
	def get_sampler_name(self):
		return self.schedule
	def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T, img_callback=None, log_every_t=None):
		sigmas = self.model_wrap.get_sigmas(S)
		x = x_T * sigmas[0]
		model_wrap_cfg = CFGDenoiser(self.model_wrap)
		samples_ddim = None
		samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas,
                                                                              extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,
                                                                                          'cond_scale': unconditional_guidance_scale}, disable=False, callback=generation_callback)
		#
		return samples_ddim, None


@torch.no_grad()
def log_likelihood(model, x, sigma_min, sigma_max, extra_args=None, atol=1e-4, rtol=1e-4):
	extra_args = {} if extra_args is None else extra_args
	s_in = x.new_ones([x.shape[0]])
	v = torch.randint_like(x, 2) * 2 - 1
	fevals = 0
	def ode_fn(sigma, x):
		nonlocal fevals
		with torch.enable_grad():
			x = x[0].detach().requires_grad_()
			denoised = model(x, sigma * s_in, **extra_args)
			d = to_d(x, sigma, denoised)
			fevals += 1
			grad = torch.autograd.grad((d * v).sum(), x)[0]
			d_ll = (v * grad).flatten(1).sum(1)
		return d.detach(), d_ll
	x_min = x, x.new_zeros([x.shape[0]])
	t = x.new_tensor([sigma_min, sigma_max])
	sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
	latent, delta_ll = sol[0][-1], sol[1][-1]
	ll_prior = torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
	return ll_prior + delta_ll, {'fevals': fevals}


def create_random_tensors(shape, seeds):
	xs = []
	for seed in seeds:
		torch.manual_seed(seed)

		# randn results depend on device; gpu and cpu get different results for same seed;
		# the way I see it, it's better to do this on CPU, so that everyone gets same result;
		# but the original script had it like this so i do not dare change it for now because
		# it will break everyone's seeds.
		xs.append(torch.randn(shape, device=defaults.general.gpu))
	x = torch.stack(xs)
	return x

def torch_gc():
	torch.cuda.empty_cache()
	torch.cuda.ipc_collect()

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

prompt_parser = re.compile("""
    (?P<prompt>                # capture group for 'prompt'
    [^:]+                      # match one or more non ':' characters
    )                          # end 'prompt'
    (?:                        # non-capture group
    :+                         # match one or more ':' characters  
    (?P<weight>                # capture group for 'weight'
    -?\\d+(?:\\.\\d+)?            # match positive or negative decimal number
    )?                         # end weight capture group, make optional 
    \\s*                        # strip spaces after weight
    |                          # OR
    $                          # else, if no ':' then match end of line
    )                          # end non-capture group
""", re.VERBOSE)

# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
def split_weighted_subprompts(input_string, normalize=True):
	parsed_prompts = [(match.group("prompt"), float(match.group("weight") or 1)) for match in re.finditer(prompt_parser, input_string)]
	if not normalize:
		return parsed_prompts
	# this probably still doesn't handle negative weights very well
	weight_sum = sum(map(lambda x: x[1], parsed_prompts))
	return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

def slerp(device, t, v0:torch.Tensor, v1:torch.Tensor, DOT_THRESHOLD=0.9995):
	v0 = v0.detach().cpu().numpy()
	v1 = v1.detach().cpu().numpy()

	dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
	if np.abs(dot) > DOT_THRESHOLD:
		v2 = (1 - t) * v0 + t * v1
	else:
		theta_0 = np.arccos(dot)
		sin_theta_0 = np.sin(theta_0)
		theta_t = theta_0 * t
		sin_theta_t = np.sin(theta_t)
		s0 = np.sin(theta_0 - theta_t) / sin_theta_0
		s1 = sin_theta_t / sin_theta_0
		v2 = s0 * v0 + s1 * v1

	v2 = torch.from_numpy(v2).to(device)

	return v2


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



def get_font(fontsize):
	fonts = ["arial.ttf", "DejaVuSans.ttf"]
	for font_name in fonts:
		try:
			return ImageFont.truetype(font_name, fontsize)
		except OSError:
			pass

	# ImageFont.load_default() is practically unusable as it only supports
	# latin1, so raise an exception instead if no usable font was found
	raise Exception(f"No usable font found (tried {', '.join(fonts)})")

def load_embeddings(fp):
	if fp is not None and hasattr(st.session_state["model"], "embedding_manager"):
		st.session_state["model"].embedding_manager.load(fp['name'])

def image_grid(imgs, batch_size, force_n_rows=None, captions=None):
	#print (len(imgs))
	if force_n_rows is not None:
		rows = force_n_rows
	elif defaults.general.n_rows > 0:
		rows = defaults.general.n_rows
	elif defaults.general.n_rows == 0:
		rows = batch_size
	else:
		rows = math.sqrt(len(imgs))
		rows = round(rows)

	cols = math.ceil(len(imgs) / rows)

	w, h = imgs[0].size
	grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

	fnt = get_font(30)

	for i, img in enumerate(imgs):
		grid.paste(img, box=(i % cols * w, i // cols * h))
		if captions and i<len(captions):
			d = ImageDraw.Draw( grid )
			size = d.textbbox( (0,0), captions[i], font=fnt, stroke_width=2, align="center" )
			d.multiline_text((i % cols * w + w/2, i // cols * h + h - size[3]), captions[i], font=fnt, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0), anchor="mm", align="center")

	return grid

def seed_to_int(s):
	if type(s) is int:
		return s
	if s is None or s == '':
		return random.randint(0, 2**32 - 1)
	n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
	while n >= 2**32:
		n = n >> 32
	return n

def check_prompt_length(prompt, comments):
	"""this function tests if prompt is too long, and if so, adds a message to comments"""

	tokenizer = (st.session_state["model"] if not defaults.general.optimized else modelCS).cond_stage_model.tokenizer
	max_length = (st.session_state["model"] if not defaults.general.optimized else modelCS).cond_stage_model.max_length

	info = (st.session_state["model"] if not defaults.general.optimized else modelCS).cond_stage_model.tokenizer([prompt], truncation=True, max_length=max_length,
                                                                                                                     return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
	ovf = info['overflowing_tokens'][0]
	overflowing_count = ovf.shape[0]
	if overflowing_count == 0:
		return

	vocab = {v: k for k, v in tokenizer.get_vocab().items()}
	overflowing_words = [vocab.get(int(x), "") for x in ovf]
	overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

	comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

def save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images):

	filename_i = os.path.join(sample_path_i, filename)

	if not jpg_sample:
		if defaults.general.save_metadata:
			metadata = PngInfo()
			metadata.add_text("SD:prompt", prompts[i])
			metadata.add_text("SD:seed", str(seeds[i]))
			metadata.add_text("SD:width", str(width))
			metadata.add_text("SD:height", str(height))
			metadata.add_text("SD:steps", str(steps))
			metadata.add_text("SD:cfg_scale", str(cfg_scale))
			metadata.add_text("SD:normalize_prompt_weights", str(normalize_prompt_weights))
			if init_img is not None:
				metadata.add_text("SD:denoising_strength", str(denoising_strength))
			metadata.add_text("SD:GFPGAN", str(use_GFPGAN and st.session_state["GFPGAN"] is not None))
			image.save(f"{filename_i}.png", pnginfo=metadata)
		else:
			image.save(f"{filename_i}.png")
	else:
		image.save(f"{filename_i}.jpg", 'jpeg', quality=100, optimize=True)

	if write_info_files:
		# toggles differ for txt2img vs. img2img:
		offset = 0 if init_img is None else 2
		toggles = []
		if prompt_matrix:
			toggles.append(0)
		if normalize_prompt_weights:
			toggles.append(1)
		if init_img is not None:
			if uses_loopback:
				toggles.append(2)
			if uses_random_seed_loopback:
				toggles.append(3)
		if save_individual_images:
			toggles.append(2 + offset)
		if save_grid:
			toggles.append(3 + offset)
		if sort_samples:
			toggles.append(4 + offset)
		if write_info_files:
			toggles.append(5 + offset)
		if use_GFPGAN:
			toggles.append(6 + offset)
		info_dict = dict(
                        target="txt2img" if init_img is None else "img2img",
                prompt=prompts[i], ddim_steps=steps, toggles=toggles, sampler_name=sampler_name,
                    ddim_eta=ddim_eta, n_iter=n_iter, batch_size=batch_size, cfg_scale=cfg_scale,
                        seed=seeds[i], width=width, height=height
                )
		if init_img is not None:
			# Not yet any use for these, but they bloat up the files:
			#info_dict["init_img"] = init_img
			#info_dict["init_mask"] = init_mask
			info_dict["denoising_strength"] = denoising_strength
			info_dict["resize_mode"] = resize_mode
		with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
			yaml.dump(info_dict, f, allow_unicode=True, width=10000)

	# render the image on the frontend
	st.session_state["preview_image"].image(image)    

def get_next_sequence_number(path, prefix=''):
	"""
	Determines and returns the next sequence number to use when saving an
	image in the specified directory.

	If a prefix is given, only consider files whose names start with that
	prefix, and strip the prefix from filenames before extracting their
	sequence number.

	The sequence starts at 0.
	"""
	result = -1
	for p in Path(path).iterdir():
		if p.name.endswith(('.png', '.jpg')) and p.name.startswith(prefix):
			tmp = p.name[len(prefix):]
			try:
				result = max(int(tmp.split('-')[0]), result)
			except ValueError:
				pass
	return result + 1


def oxlamon_matrix(prompt, seed, n_iter, batch_size):
	pattern = re.compile(r'(,\s){2,}')

	class PromptItem:
		def __init__(self, text, parts, item):
			self.text = text
			self.parts = parts
			if item:
				self.parts.append( item )

	def clean(txt):
		return re.sub(pattern, ', ', txt)

	def getrowcount( txt ):
		for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
			if data:
				return len(data.group(1).split("|"))
			break
		return None

	def repliter( txt ):
		for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
			if data:
				r = data.span(1)
				for item in data.group(1).split("|"):
					yield (clean(txt[:r[0]-1] + item.strip() + txt[r[1]+1:]), item.strip())
			break

	def iterlist( items ):
		outitems = []
		for item in items:
			for newitem, newpart in repliter(item.text):
				outitems.append( PromptItem(newitem, item.parts.copy(), newpart) )

		return outitems

	def getmatrix( prompt ):
		dataitems = [ PromptItem( prompt[1:].strip(), [], None ) ]
		while True:
			newdataitems = iterlist( dataitems )
			if len( newdataitems ) == 0:
				return dataitems
			dataitems = newdataitems

	def classToArrays( items, seed, n_iter ):
		texts = []
		parts = []
		seeds = []

		for item in items:
			itemseed = seed
			for i in range(n_iter):
				texts.append( item.text )
				parts.append( f"Seed: {itemseed}\n" + "\n".join(item.parts) )
				seeds.append( itemseed )
				itemseed += 1                

		return seeds, texts, parts

	all_seeds, all_prompts, prompt_matrix_parts = classToArrays(getmatrix( prompt ), seed, n_iter)
	n_iter = math.ceil(len(all_prompts) / batch_size)

	needrows = getrowcount(prompt)
	if needrows:
		xrows = math.sqrt(len(all_prompts))
		xrows = round(xrows)
		# if columns is to much
		cols = math.ceil(len(all_prompts) / xrows)
		if cols > needrows*4:
			needrows *= 2

	return all_seeds, n_iter, prompt_matrix_parts, all_prompts, needrows


def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, save_grid, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        fp=None, ddim_eta=0.0, normalize_prompt_weights=True, init_img=None, init_mask=None,
        keep_mask=False, mask_blur_strength=3, denoising_strength=0.75, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, jpg_sample=False,
        variant_amount=0.0, variant_seed=None, save_individual_images: bool = True):
	"""this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
	assert prompt is not None
	torch_gc()
	# start time after garbage collection (or before?)
	start_time = time.time()

	# We will use this date here later for the folder name, need to start_time if not need
	run_start_dt = datetime.datetime.now()

	mem_mon = MemUsageMonitor('MemMon')
	mem_mon.start()

	if hasattr(st.session_state["model"], "embedding_manager"):
		load_embeddings(fp)

	os.makedirs(outpath, exist_ok=True)

	sample_path = os.path.join(outpath, "samples")
	os.makedirs(sample_path, exist_ok=True)

	if not ("|" in prompt) and prompt.startswith("@"):
		prompt = prompt[1:]

	comments = []

	prompt_matrix_parts = []
	simple_templating = False
	add_original_image = not (use_RealESRGAN or use_GFPGAN)

	if prompt_matrix:
		if prompt.startswith("@"):
			simple_templating = True
			add_original_image = not (use_RealESRGAN or use_GFPGAN)
			all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(prompt, seed, n_iter, batch_size)
		else:
			all_prompts = []
			prompt_matrix_parts = prompt.split("|")
			combination_count = 2 ** (len(prompt_matrix_parts) - 1)
			for combination_num in range(combination_count):
				current = prompt_matrix_parts[0]

				for n, text in enumerate(prompt_matrix_parts[1:]):
					if combination_num & (2 ** n) > 0:
						current += ("" if text.strip().startswith(",") else ", ") + text

				all_prompts.append(current)

			n_iter = math.ceil(len(all_prompts) / batch_size)
			all_seeds = len(all_prompts) * [seed]

		print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
	else:

		if not defaults.general.no_verify_input:
			try:
				check_prompt_length(prompt, comments)
			except:
				import traceback
				print("Error verifying input:", file=sys.stderr)
				print(traceback.format_exc(), file=sys.stderr)

		all_prompts = batch_size * n_iter * [prompt]
		all_seeds = [seed + x for x in range(len(all_prompts))]

	precision_scope = autocast if defaults.general.precision == "autocast" else nullcontext
	output_images = []
	grid_captions = []
	stats = []
	with torch.no_grad(), precision_scope("cuda"), (st.session_state["model"].ema_scope() if not defaults.general.optimized else nullcontext()):
		init_data = func_init()
		tic = time.time()


		# if variant_amount > 0.0 create noise from base seed
		base_x = None
		if variant_amount > 0.0:
			target_seed_randomizer = seed_to_int('') # random seed
			torch.manual_seed(seed) # this has to be the single starting seed (not per-iteration)
			base_x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=[seed])
			# we don't want all_seeds to be sequential from starting seed with variants, 
			# since that makes the same variants each time, 
			# so we add target_seed_randomizer as a random offset 
			for si in range(len(all_seeds)):
				all_seeds[si] += target_seed_randomizer

		for n in range(n_iter):
			print(f"Iteration: {n+1}/{n_iter}")
			prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
			captions = prompt_matrix_parts[n * batch_size:(n + 1) * batch_size]
			seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

			print(prompt)

			if defaults.general.optimized:
				modelCS.to(defaults.general.gpu)

			uc = (st.session_state["model"] if not defaults.general.optimized else modelCS).get_learned_conditioning(len(prompts) * [""])

			if isinstance(prompts, tuple):
				prompts = list(prompts)

			# split the prompt if it has : for weighting
			# TODO for speed it might help to have this occur when all_prompts filled??
			weighted_subprompts = split_weighted_subprompts(prompts[0], normalize_prompt_weights)

			# sub-prompt weighting used if more than 1
			if len(weighted_subprompts) > 1:
				c = torch.zeros_like(uc) # i dont know if this is correct.. but it works
				for i in range(0, len(weighted_subprompts)):
					# note if alpha negative, it functions same as torch.sub
					c = torch.add(c, (st.session_state["model"] if not defaults.general.optimized else modelCS).get_learned_conditioning(weighted_subprompts[i][0]), alpha=weighted_subprompts[i][1])
			else: # just behave like usual
				c = (st.session_state["model"] if not defaults.general.optimized else modelCS).get_learned_conditioning(prompts)


			shape = [opt_C, height // opt_f, width // opt_f]

			if defaults.general.optimized:
				mem = torch.cuda.memory_allocated()/1e6
				modelCS.to("cpu")
				while(torch.cuda.memory_allocated()/1e6 >= mem):
					time.sleep(1)

			if variant_amount == 0.0:
				# we manually generate all input noises because each one should have a specific seed
				x = create_random_tensors(shape, seeds=seeds)

			else: # we are making variants
				# using variant_seed as sneaky toggle, 
				# when not None or '' use the variant_seed
				# otherwise use seeds
				if variant_seed != None and variant_seed != '':
					specified_variant_seed = seed_to_int(variant_seed)
					torch.manual_seed(specified_variant_seed)
					seeds = [specified_variant_seed]
				target_x = create_random_tensors(shape, seeds=seeds)
				# finally, slerp base_x noise to target_x noise for creating a variant
				x = slerp(defaults.general.gpu, max(0.0, min(1.0, variant_amount)), base_x, target_x)

			samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

			if defaults.general.optimized:
				modelFS.to(defaults.general.gpu)

			x_samples_ddim = (st.session_state["model"] if not defaults.general.optimized else modelFS).decode_first_stage(samples_ddim)
			x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

			for i, x_sample in enumerate(x_samples_ddim):				
				sanitized_prompt = slugify(prompts[i])

				if sort_samples:
					full_path = os.path.join(os.getcwd(), sample_path, sanitized_prompt)


					sanitized_prompt = sanitized_prompt[:220-len(full_path)]
					sample_path_i = os.path.join(sample_path, sanitized_prompt)

					#print(f"output folder length: {len(os.path.join(os.getcwd(), sample_path_i))}")
					#print(os.path.join(os.getcwd(), sample_path_i))

					os.makedirs(sample_path_i, exist_ok=True)
					base_count = get_next_sequence_number(sample_path_i)
					filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}"
				else:
					full_path = os.path.join(os.getcwd(), sample_path)
					sample_path_i = sample_path
					base_count = get_next_sequence_number(sample_path_i)
					filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[:220-len(full_path)] #same as before

				x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
				x_sample = x_sample.astype(np.uint8)
				image = Image.fromarray(x_sample)
				original_sample = x_sample
				original_filename = filename

				if use_GFPGAN and st.session_state["GFPGAN"] is not None and not use_RealESRGAN:
					#skip_save = True # #287 >_>
					torch_gc()
					cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
					gfpgan_sample = restored_img[:,:,::-1]
					gfpgan_image = Image.fromarray(gfpgan_sample)
					gfpgan_filename = original_filename + '-gfpgan'

					save_sample(gfpgan_image, sample_path_i, gfpgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback,
                                                    uses_random_seed_loopback, save_grid, sort_samples, sampler_name, ddim_eta,
                                                    n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images=False)

					output_images.append(gfpgan_image) #287
					if simple_templating:
						grid_captions.append( captions[i] + "\ngfpgan" )

				if use_RealESRGAN and st.session_state["RealESRGAN"] is not None and not use_GFPGAN:
					#skip_save = True # #287 >_>
					torch_gc()

					if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
						#try_loading_RealESRGAN(realesrgan_model_name)
						load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

					output, img_mode = st.session_state["RealESRGAN"].enhance(x_sample[:,:,::-1])
					esrgan_filename = original_filename + '-esrgan4x'
					esrgan_sample = output[:,:,::-1]
					esrgan_image = Image.fromarray(esrgan_sample)

					#save_sample(image, sample_path_i, original_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
							#normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
							#save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)

					save_sample(esrgan_image, sample_path_i, esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                                    save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images=False)

					output_images.append(esrgan_image) #287
					if simple_templating:
						grid_captions.append( captions[i] + "\nesrgan" )

				if use_RealESRGAN and st.session_state["RealESRGAN"] is not None and use_GFPGAN and st.session_state["GFPGAN"] is not None:
					#skip_save = True # #287 >_>
					torch_gc()
					cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
					gfpgan_sample = restored_img[:,:,::-1]

					if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
						#try_loading_RealESRGAN(realesrgan_model_name)
						load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

					output, img_mode = st.session_state["RealESRGAN"].enhance(gfpgan_sample[:,:,::-1])
					gfpgan_esrgan_filename = original_filename + '-gfpgan-esrgan4x'
					gfpgan_esrgan_sample = output[:,:,::-1]
					gfpgan_esrgan_image = Image.fromarray(gfpgan_esrgan_sample)						    

					save_sample(gfpgan_esrgan_image, sample_path_i, gfpgan_esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, False, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                                    save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images=False)

					output_images.append(gfpgan_esrgan_image) #287

					if simple_templating:
						grid_captions.append( captions[i] + "\ngfpgan_esrgan" )

				if save_individual_images:
					save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                                    save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images)

					if not use_GFPGAN or not use_RealESRGAN:
						output_images.append(image)

					#if add_original_image or not simple_templating:
						#output_images.append(image)
						#if simple_templating:
							#grid_captions.append( captions[i] )

				if defaults.general.optimized:
					mem = torch.cuda.memory_allocated()/1e6
					modelFS.to("cpu")
					while(torch.cuda.memory_allocated()/1e6 >= mem):
						time.sleep(1)

		if prompt_matrix or save_grid:
			if prompt_matrix:
				if simple_templating:
					grid = image_grid(output_images, n_iter, force_n_rows=frows, captions=grid_captions)
				else:
					grid = image_grid(output_images, n_iter, force_n_rows=1 << ((len(prompt_matrix_parts)-1)//2))
					try:
						grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
					except:
						import traceback
						print("Error creating prompt_matrix text:", file=sys.stderr)
						print(traceback.format_exc(), file=sys.stderr)
			else:
				grid = image_grid(output_images, batch_size)

			if grid and (batch_size > 1  or n_iter > 1):
				output_images.insert(0, grid)

			grid_count = get_next_sequence_number(outpath, 'grid-')
			grid_file = f"grid-{grid_count:05}-{seed}_{slugify(prompts[i].replace(' ', '_')[:220-len(full_path)])}.{grid_ext}"
			grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)

		toc = time.time()

	mem_max_used, mem_total = mem_mon.read_and_stop()
	time_diff = time.time()-start_time

	info = f"""
                {prompt}
                Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', Denoising strength: '+str(denoising_strength) if init_img is not None else ''}{', GFPGAN' if use_GFPGAN and st.session_state["GFPGAN"] is not None else ''}{', '+realesrgan_model_name if use_RealESRGAN and st.session_state["RealESRGAN"] is not None else ''}{', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()
	stats = f'''
                Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
                Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

	for comment in comments:
		info += "\n\n" + comment

	#mem_mon.stop()
	#del mem_mon
	torch_gc()

	return output_images, seed, info, stats


def resize_image(resize_mode, im, width, height):
	LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
	if resize_mode == 0:
		res = im.resize((width, height), resample=LANCZOS)
	elif resize_mode == 1:
		ratio = width / height
		src_ratio = im.width / im.height

		src_w = width if ratio > src_ratio else im.width * height // im.height
		src_h = height if ratio <= src_ratio else im.height * width // im.width

		resized = im.resize((src_w, src_h), resample=LANCZOS)
		res = Image.new("RGBA", (width, height))
		res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
	else:
		ratio = width / height
		src_ratio = im.width / im.height

		src_w = width if ratio < src_ratio else im.width * height // im.height
		src_h = height if ratio >= src_ratio else im.height * width // im.width

		resized = im.resize((src_w, src_h), resample=LANCZOS)
		res = Image.new("RGBA", (width, height))
		res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

		if ratio < src_ratio:
			fill_height = height // 2 - src_h // 2
			res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
			res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
		elif ratio > src_ratio:
			fill_width = width // 2 - src_w // 2
			res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
			res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

	return res

def img2img(prompt: str = '', init_info: any = None, init_info_mask: any = None, mask_mode: int = 0, mask_blur_strength: int = 3, 
            ddim_steps: int = 50, sampler_name: str = 'DDIM',
            n_iter: int = 1,  cfg_scale: float = 7.5, denoising_strength: float = 0.8,
            seed: int = -1, height: int = 512, width: int = 512, resize_mode: int = 0, fp = None,
            variant_amount: float = None, variant_seed: int = None, ddim_eta:float = 0.0,
            write_info_files:bool = True, RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B",
            separate_prompts:bool = False, normalize_prompt_weights:bool = True,
            save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
            save_as_jpg: bool = True, use_GFPGAN: bool = True, use_RealESRGAN: bool = True, loopback: bool = False,
            random_seed_loopback: bool = False
            ):

	outpath = defaults.general.outdir_img2img or defaults.general.outdir or "outputs/img2img-samples"
	err = False
	#loopback = False
	#skip_save = False
	seed = seed_to_int(seed)

	batch_size = 1

	#prompt_matrix = 0
	#normalize_prompt_weights = 1 in toggles
	#loopback = 2 in toggles
	#random_seed_loopback = 3 in toggles
	#skip_save = 4 not in toggles
	#save_grid = 5 in toggles
	#sort_samples = 6 in toggles
	#write_info_files = 7 in toggles
	#write_sample_info_to_log_file = 8 in toggles
	#jpg_sample = 9 in toggles
	#use_GFPGAN = 10 in toggles
	#use_RealESRGAN = 11 in toggles

	if sampler_name == 'PLMS':
		sampler = PLMSSampler(st.session_state["model"])
	elif sampler_name == 'DDIM':
		sampler = DDIMSampler(st.session_state["model"])
	elif sampler_name == 'k_dpm_2_a':
		sampler = KDiffusionSampler(st.session_state["model"],'dpm_2_ancestral')
	elif sampler_name == 'k_dpm_2':
		sampler = KDiffusionSampler(st.session_state["model"],'dpm_2')
	elif sampler_name == 'k_euler_a':
		sampler = KDiffusionSampler(st.session_state["model"],'euler_ancestral')
	elif sampler_name == 'k_euler':
		sampler = KDiffusionSampler(st.session_state["model"],'euler')
	elif sampler_name == 'k_heun':
		sampler = KDiffusionSampler(st.session_state["model"],'heun')
	elif sampler_name == 'k_lms':
		sampler = KDiffusionSampler(st.session_state["model"],'lms')
	else:
		raise Exception("Unknown sampler: " + sampler_name)

	init_img = init_info
	init_mask = None
	keep_mask = False

	assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
	t_enc = int(denoising_strength * ddim_steps)

	def init():

		image = init_img
		image = np.array(image).astype(np.float32) / 255.0
		image = image[None].transpose(0, 3, 1, 2)
		image = torch.from_numpy(image)

		mask = None
		if defaults.general.optimized:
			modelFS.to(st.session_state["device"] )

		init_image = 2. * image - 1.
		init_image = init_image.to(st.session_state["device"])
		init_latent = (st.session_state["model"] if not defaults.general.optimized else modelFS).get_first_stage_encoding((st.session_state["model"]  if not defaults.general.optimized else modelFS).encode_first_stage(init_image))  # move to latent space

		if defaults.general.optimized:
			mem = torch.cuda.memory_allocated()/1e6
			modelFS.to("cpu")
			while(torch.cuda.memory_allocated()/1e6 >= mem):
				time.sleep(1)

		return init_latent, mask,

	def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
		t_enc_steps = t_enc
		obliterate = False
		if ddim_steps == t_enc_steps:
			t_enc_steps = t_enc_steps - 1
			obliterate = True

		if sampler_name != 'DDIM':
			x0, z_mask = init_data

			sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
			noise = x * sigmas[ddim_steps - t_enc_steps - 1]

			xi = x0 + noise

			# Obliterate masked image
			if z_mask is not None and obliterate:
				random = torch.randn(z_mask.shape, device=xi.device)
				xi = (z_mask * noise) + ((1-z_mask) * xi)

			sigma_sched = sigmas[ddim_steps - t_enc_steps - 1:]
			model_wrap_cfg = CFGMaskedDenoiser(sampler.model_wrap)
			samples_ddim = K.sampling.__dict__[f'sample_{sampler.get_sampler_name()}'](model_wrap_cfg, xi, sigma_sched,
                                                                                                   extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,
                                                                                                               'cond_scale': cfg_scale, 'mask': z_mask, 'x0': x0, 'xi': xi}, disable=False,
                                                                                                   callback=generation_callback)
		else:

			x0, z_mask = init_data

			sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
			z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc_steps]*batch_size).to(st.session_state["device"] ))

			# Obliterate masked image
			if z_mask is not None and obliterate:
				random = torch.randn(z_mask.shape, device=z_enc.device)
				z_enc = (z_mask * random) + ((1-z_mask) * z_enc)

								# decode it
			samples_ddim = sampler.decode(z_enc, conditioning, t_enc_steps,
                                                      unconditional_guidance_scale=cfg_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                            z_mask=z_mask, x0=x0)
		return samples_ddim



	if loopback:
		output_images, info = None, None
		history = []
		initial_seed = None

		do_color_correction = False
		try:
			from skimage import exposure
			do_color_correction = True
		except:
			print("Install scikit-image to perform color correction on loopback")		

		for i in range(1):
			if do_color_correction and i == 0:
				correction_target = cv2.cvtColor(np.asarray(init_img.copy()), cv2.COLOR_RGB2LAB)

			output_images, seed, info, stats = process_images(
                                outpath=outpath,
                                func_init=init,
                                func_sample=sample,
                                prompt=prompt,
                                seed=seed,
                                sampler_name=sampler_name,
                                save_grid=save_grid,
                                batch_size=1,
                                n_iter=n_iter,
                                steps=ddim_steps,
                                cfg_scale=cfg_scale,
                                width=width,
                                height=height,
                                prompt_matrix=separate_prompts,
                                use_GFPGAN=use_GFPGAN,
                                use_RealESRGAN=use_RealESRGAN, # Forcefully disable upscaling when using loopback
                                realesrgan_model_name=RealESRGAN_model,
                                fp=fp,
                                normalize_prompt_weights=normalize_prompt_weights,
                                save_individual_images=save_individual_images,
                                init_img=init_img,
                                init_mask=init_mask,
                                keep_mask=keep_mask,
                                mask_blur_strength=mask_blur_strength,
                                denoising_strength=denoising_strength,
                                resize_mode=resize_mode,
                                uses_loopback=loopback,
                                uses_random_seed_loopback=random_seed_loopback,
                                sort_samples=group_by_prompt,
                                write_info_files=write_info_files,
                                jpg_sample=save_as_jpg
                        )

			if initial_seed is None:
				initial_seed = seed

			init_img = output_images[0]

			if do_color_correction and correction_target is not None:
				init_img = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
                                        cv2.cvtColor(
                                                np.asarray(init_img),
                                            cv2.COLOR_RGB2LAB
                                        ),
                                        correction_target,
                                    channel_axis=2
                                    ), cv2.COLOR_LAB2RGB).astype("uint8"))

			if not random_seed_loopback:
				seed = seed + 1
			else:
				seed = seed_to_int(None)

			denoising_strength = max(denoising_strength * 0.95, 0.1)
			history.append(init_img)

		output_images = history
		seed = initial_seed

	else:
		output_images, seed, info, stats = process_images(
                        outpath=outpath,
                        func_init=init,
                        func_sample=sample,
                        prompt=prompt,
                        seed=seed,
                        sampler_name=sampler_name,
                        save_grid=save_grid,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        steps=ddim_steps,
                        cfg_scale=cfg_scale,
                        width=width,
                        height=height,
                        prompt_matrix=separate_prompts,
                        use_GFPGAN=use_GFPGAN,
                        use_RealESRGAN=use_RealESRGAN,
                        realesrgan_model_name=RealESRGAN_model,
                        fp=fp,
                        normalize_prompt_weights=normalize_prompt_weights,
                        save_individual_images=save_individual_images,
                        init_img=init_img,
                        init_mask=init_mask,
                        keep_mask=keep_mask,
                        mask_blur_strength=2,
                        denoising_strength=denoising_strength,
                        resize_mode=resize_mode,
                        uses_loopback=loopback,
                        sort_samples=group_by_prompt,
                        write_info_files=write_info_files,
                        jpg_sample=save_as_jpg
                )

	del sampler

	return output_images, seed, info, stats

#@retry(RuntimeError, tries=3)
def txt2img(prompt: str, ddim_steps: int, sampler_name: str, realesrgan_model_name: str,
            n_iter: int, batch_size: int, cfg_scale: float, seed: Union[int, str, None],
            height: int, width: int, separate_prompts:bool = False, normalize_prompt_weights:bool = True,
            save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
            save_as_jpg: bool = True, use_GFPGAN: bool = True, use_RealESRGAN: bool = True, 
            RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B", fp = None, variant_amount: float = None, 
            variant_seed: int = None, ddim_eta:float = 0.0, write_info_files:bool = True):

	outpath = defaults.general.outdir_txt2img or defaults.general.outdir or "outputs/txt2img-samples"

	err = False
	seed = seed_to_int(seed)

	#prompt_matrix = 0 in toggles
	#normalize_prompt_weights = 1 in toggles
	#skip_save = 2 not in toggles
	#save_grid = 3 not in toggles
	#sort_samples = 4 in toggles
	#write_info_files = 5 in toggles
	#jpg_sample = 6 in toggles
	#use_GFPGAN = 7 in toggles
	#use_RealESRGAN = 8 in toggles

	if sampler_name == 'PLMS':
		sampler = PLMSSampler(st.session_state["model"])
	elif sampler_name == 'DDIM':
		sampler = DDIMSampler(st.session_state["model"])
	elif sampler_name == 'k_dpm_2_a':
		sampler = KDiffusionSampler(st.session_state["model"],'dpm_2_ancestral')
	elif sampler_name == 'k_dpm_2':
		sampler = KDiffusionSampler(st.session_state["model"],'dpm_2')
	elif sampler_name == 'k_euler_a':
		sampler = KDiffusionSampler(st.session_state["model"],'euler_ancestral')
	elif sampler_name == 'k_euler':
		sampler = KDiffusionSampler(st.session_state["model"],'euler')
	elif sampler_name == 'k_heun':
		sampler = KDiffusionSampler(st.session_state["model"],'heun')
	elif sampler_name == 'k_lms':
		sampler = KDiffusionSampler(st.session_state["model"],'lms')
	else:
		raise Exception("Unknown sampler: " + sampler_name)

	def init():
		pass

	def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
		samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale,
                                                 unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=generation_callback,
                                         log_every_t=int(defaults.general.update_preview_frequency))

		return samples_ddim

	#try:
	output_images, seed, info, stats = process_images(
                outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_name=sampler_name,
                save_grid=save_grid,
                batch_size=batch_size,
                n_iter=n_iter,      
                steps=ddim_steps,   
                cfg_scale=cfg_scale,	
                width=width,
                height=height,
                prompt_matrix=separate_prompts,
                use_GFPGAN=use_GFPGAN,
                use_RealESRGAN=use_RealESRGAN,
                realesrgan_model_name=realesrgan_model_name,
                fp=fp,
                ddim_eta=ddim_eta,
                normalize_prompt_weights=normalize_prompt_weights,
                save_individual_images=save_individual_images,
                sort_samples=group_by_prompt,
                write_info_files=write_info_files,
                jpg_sample=save_as_jpg,
                variant_amount=variant_amount,
                variant_seed=variant_seed,
        )

	del sampler

	return output_images, seed, info, stats

	#except RuntimeError as e:
		#err = e
		#err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
		#stats = err_msg
		#return [], seed, 'err', stats




# functions to load css locally OR remotely starts here. Options exist for future flexibility. Called as st.markdown with unsafe_allow_html as css injection
# TODO, maybe look into async loading the file especially for remote fetching 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def load_css(isLocal, nameOrURL):
	if(isLocal):
		local_css(nameOrURL)
	else:
		remote_css(nameOrURL)


# main functions to define streamlit layout here
def layout():

	st.set_page_config(page_title="Stable Diffusion Playground", layout="wide", initial_sidebar_state="collapsed")

	with st.empty():
		# load css as an external file, function has an option to local or remote url. Potential use when running from cloud infra that might not have access to local path.
		load_css(True, 'frontend/css/streamlit.main.css')

	# check if the models exist on their respective folders
	if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
		GFPGAN_available = True
	else:
		GFPGAN_available = False

	if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{defaults.general.RealESRGAN_model}.pth")):
		RealESRGAN_available = True
	else:
		RealESRGAN_available = False	

	with st.sidebar:
		# we should use an expander and group things together when more options are added so the sidebar is not too messy.
		#with st.expander("Global Settings:"):
		st.write("Global Settings:")
		defaults.general.update_preview = st.checkbox("Update Image Preview", value=defaults.general.update_preview,
                                                              help="If enabled the image preview will be updated during the generation instead of at the end. You can use the Update Preview \
							      Frequency option bellow to customize how frequent it's updated. By default this is enabled and the frequency is set to 1 step.")
		defaults.general.update_preview_frequency = st.text_input("Update Image Preview Frequency", value=defaults.general.update_preview_frequency,
                                                                          help="Frequency in steps at which the the preview image is updated. By default the frequency is set to 1 step.")



	txt2img_tab, img2img_tab, txt2video, postprocessing_tab = st.tabs(["Text-to-Image Unified", "Image-to-Image Unified", "Text-to-Video","Post-Processing"])

	with txt2img_tab:		
		with st.form("txt2img-inputs"):
			st.session_state["generation_mode"] = "txt2img"

			input_col1, generate_col1 = st.columns([10,1])
			with input_col1:
				#prompt = st.text_area("Input Text","")
				prompt = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")

			# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
			generate_col1.write("")
			generate_col1.write("")
			generate_button = generate_col1.form_submit_button("Generate")

			# creating the page layout using columns
			col1, col2, col3 = st.columns([1,2,1], gap="large")    

			with col1:
				width = st.slider("Width:", min_value=64, max_value=1024, value=defaults.txt2img.width, step=64)
				height = st.slider("Height:", min_value=64, max_value=1024, value=defaults.txt2img.height, step=64)
				cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.txt2img.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")
				seed = st.text_input("Seed:", value=defaults.txt2img.seed, help=" The seed to use, if left blank a random seed will be generated.")
				batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.txt2img.batch_count, step=1, help="How many iterations or batches of images to generate in total.")
				#batch_size = st.slider("Batch size", min_value=1, max_value=250, value=defaults.txt2img.batch_size, step=1,
					#help="How many images are at once in a batch.\
					#It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
					#Default: 1")

			with col2:
				preview_tab, gallery_tab = st.tabs(["Preview", "Gallery"])

				with preview_tab:
					#st.write("Image")
					#Image for testing
					#image = Image.open(requests.get("https://icon-library.com/images/image-placeholder-icon/image-placeholder-icon-13.jpg", stream=True).raw).convert('RGB')
					#new_image = image.resize((175, 240))
					#preview_image = st.image(image)

					# create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
					st.session_state["preview_image"] = st.empty()

					st.session_state["loading"] = st.empty()

					st.session_state["progress_bar_text"] = st.empty()
					st.session_state["progress_bar"] = st.empty()

					message = st.empty()

				with gallery_tab:
					st.write('Here should be the image gallery, if I could make a grid in streamlit.')

			with col3:
				st.session_state.sampling_steps = st.slider("Sampling Steps", value=defaults.txt2img.sampling_steps, min_value=1, max_value=250)
				
				sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
				sampler_name = st.selectbox("Sampling method", sampler_name_list,
                                            index=sampler_name_list.index(defaults.txt2img.default_sampler), help="Sampling method to use. Default: k_euler")  



				#basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

				#with basic_tab:
					#summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
						#help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")

				with st.expander("Advanced"):
					separate_prompts = st.checkbox("Create Prompt Matrix.", value=False, help="Separate multiple prompts using the `|` character, and get all combinations of them.")
					normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=True, help="Ensure the sum of all weights add up to 1.0")
					save_individual_images = st.checkbox("Save individual images.", value=True, help="Save each image generated before any filter or enhancement is applied.")
					save_grid = st.checkbox("Save grid",value=True, help="Save a grid with all the images generated into a single image.")
					group_by_prompt = st.checkbox("Group results by prompt", value=True,
                                                                      help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
					write_info_files = st.checkbox("Write Info file", value=True, help="Save a file next to the image with informartion about the generation.")
					save_as_jpg = st.checkbox("Save samples as jpg", value=False, help="Saves the images as jpg instead of png.")

					if GFPGAN_available:
						use_GFPGAN = st.checkbox("Use GFPGAN", value=defaults.txt2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation. This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
					else:
						use_GFPGAN = False

					if RealESRGAN_available:
						use_RealESRGAN = st.checkbox("Use RealESRGAN", value=defaults.txt2img.use_RealESRGAN, help="Uses the RealESRGAN model to upscale the images after the generation. This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
						RealESRGAN_model = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
					else:
						use_RealESRGAN = False
						RealESRGAN_model = "RealESRGAN_x4plus"

					variant_amount = st.slider("Variant Amount:", value=defaults.txt2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
					variant_seed = st.text_input("Variant Seed:", value=defaults.txt2img.seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")


			if generate_button:
				#print("Loading models")
				# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.		
				load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model)                

				try:
					output_images, seed, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, RealESRGAN_model, batch_count, 1, 
										   cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
										   save_grid, group_by_prompt, save_as_jpg, use_GFPGAN, use_RealESRGAN, RealESRGAN_model, fp=defaults.general.fp,
										   variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files)
	
					message.success('Done!', icon="✅")

				except (StopException, KeyError):
					print(f"Received Streamlit StopException")

				# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
				# use the current col2 first tab to show the preview_img and update it as its generated.
				#preview_image.image(output_images)

	with img2img_tab:		
		with st.form("img2img-inputs"):
			st.session_state["generation_mode"] = "img2img"

			img2img_input_col, img2img_generate_col = st.columns([10,1])
			with img2img_input_col:
				#prompt = st.text_area("Input Text","")
				prompt = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")

			# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
			img2img_generate_col.write("")
			img2img_generate_col.write("")
			generate_button = img2img_generate_col.form_submit_button("Generate")


			# creating the page layout using columns
			col1_img2img_layout, col2_img2img_layout, col3_img2img_layout = st.columns([1,2,2], gap="small")    

			with col1_img2img_layout:
				st.session_state["sampling_steps"] = st.slider("Sampling Steps", value=defaults.img2img.sampling_steps, min_value=1, max_value=250)
				st.session_state["sampler_name"] = st.selectbox("Sampling method", ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"],
                                                                                index=0, help="Sampling method to use. Default: k_lms")  				

				uploaded_images = st.file_uploader("Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg"],
                                                                   help="Upload an image which will be used for the image to image generation."
                                                                   )

				width = st.slider("Width:", min_value=64, max_value=1024, value=defaults.img2img.width, step=64)
				height = st.slider("Height:", min_value=64, max_value=1024, value=defaults.img2img.height, step=64)
				seed = st.text_input("Seed:", value=defaults.img2img.seed, help=" The seed to use, if left blank a random seed will be generated.")
				batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.img2img.batch_count, step=1, help="How many iterations or batches of images to generate in total.")

				#			
				with st.expander("Advanced"):
					separate_prompts = st.checkbox("Create Prompt Matrix.", value=defaults.img2img.separate_prompts, help="Separate multiple prompts using the `|` character, and get all combinations of them.")
					normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=defaults.img2img.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
					loopback = st.checkbox("Loopback.", value=defaults.img2img.loopback, help="Use images from previous batch when creating next batch.")
					random_seed_loopback = st.checkbox("Random loopback seed.", value=defaults.img2img.random_seed_loopback, help="Random loopback seed")
					save_individual_images = st.checkbox("Save individual images.", value=True, help="Save each image generated before any filter or enhancement is applied.")
					save_grid = st.checkbox("Save grid",value=defaults.img2img.save_grid, help="Save a grid with all the images generated into a single image.")
					group_by_prompt = st.checkbox("Group results by prompt", value=defaults.img2img.group_by_prompt,
                                                                      help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
					write_info_files = st.checkbox("Write Info file", value=True, help="Save a file next to the image with informartion about the generation.")
					save_as_jpg = st.checkbox("Save samples as jpg", value=False, help="Saves the images as jpg instead of png.")

					if GFPGAN_available:
						use_GFPGAN = st.checkbox("Use GFPGAN", value=defaults.img2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation.\
						This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
					else:
						use_GFPGAN = False

					if RealESRGAN_available:
						use_RealESRGAN = st.checkbox("Use RealESRGAN", value=defaults.img2img.use_RealESRGAN, help="Uses the RealESRGAN model to upscale the images after the generation.\
						This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
						RealESRGAN_model = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
					else:
						use_RealESRGAN = False
						RealESRGAN_model = "RealESRGAN_x4plus"

					variant_amount = st.slider("Variant Amount:", value=defaults.img2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
					variant_seed = st.text_input("Variant Seed:", value=defaults.img2img.variant_seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")
					cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.img2img.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")
					batch_size = st.slider("Batch size", min_value=1, max_value=100, value=defaults.img2img.batch_size, step=1,
                                                               help="How many images are at once in a batch.\
								       It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
								       Default: 1")

					st.session_state["denoising_strength"] = st.slider("Denoising Strength:", value=defaults.img2img.denoising_strength, min_value=0.01, max_value=1.0, step=0.01)


			with col2_img2img_layout:
				editor_tab = st.tabs(["Editor"])

				editor_image = st.empty()
				st.session_state["editor_image"] = editor_image

				if uploaded_images:
					image = Image.open(uploaded_images).convert('RGB')
					#img_array = np.array(image) # if you want to pass it to OpenCV
					new_img = image.resize((width, height))
					st.image(new_img)


				with col3_img2img_layout:
					result_tab = st.tabs(["Result"])

					# create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
					preview_image = st.empty()
					st.session_state["preview_image"] = preview_image

					#st.session_state["loading"] = st.empty()

					st.session_state["progress_bar_text"] = st.empty()
					st.session_state["progress_bar"] = st.empty()


					message = st.empty()

					#if uploaded_images:
						#image = Image.open(uploaded_images).convert('RGB')
						##img_array = np.array(image) # if you want to pass it to OpenCV
						#new_img = image.resize((width, height))
						#st.image(new_img, use_column_width=True)


			if generate_button:
				#print("Loading models")
				# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.
				load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model)                
				if uploaded_images:
					image = Image.open(uploaded_images).convert('RGB')
					new_img = image.resize((width, height))
					#img_array = np.array(image) # if you want to pass it to OpenCV

					try:
						output_images, seed, info, stats = img2img(prompt=prompt, init_info=new_img, ddim_steps=st.session_state["sampling_steps"],
											   sampler_name=st.session_state["sampler_name"], n_iter=batch_count,
											   cfg_scale=cfg_scale, denoising_strength=st.session_state["denoising_strength"], variant_seed=variant_seed,
											   seed=seed, width=width, height=height, fp=defaults.general.fp, variant_amount=variant_amount, 
											   ddim_eta=0.0, write_info_files=write_info_files, RealESRGAN_model=RealESRGAN_model,
											   separate_prompts=separate_prompts, normalize_prompt_weights=normalize_prompt_weights,
											   save_individual_images=save_individual_images, save_grid=save_grid, 
											   group_by_prompt=group_by_prompt, save_as_jpg=save_as_jpg, use_GFPGAN=use_GFPGAN,
											   use_RealESRGAN=use_RealESRGAN if not loopback else False, loopback=loopback
											   )
	
						#show a message when the generation is complete.
						message.success('Done!', icon="✅")

					except (StopException, KeyError):
						print(f"Received Streamlit StopException")

				# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
				# use the current col2 first tab to show the preview_img and update it as its generated.
				#preview_image.image(output_images, width=750)


if __name__ == '__main__':
	layout()     
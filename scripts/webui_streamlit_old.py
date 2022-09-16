import warnings

import piexif
import piexif.helper
import json

import streamlit as st
from streamlit import StopException

#streamlit components section
from st_on_hover_tabs import on_hover_tabs

import base64, cv2
import os, sys, re, random, datetime, timeit
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from scipy import integrate
import pandas as pd
import torch
from torchdiffeq import odeint
import k_diffusion as K
import math
import mimetypes
import numpy as np
import pynvml
import threading
import time, inspect
import torch
from torch import autocast
from torchvision import transforms
import torch.nn as nn
import yaml
from typing import Union
from pathlib import Path
#from tqdm import tqdm
from contextlib import nullcontext
from einops import rearrange
from omegaconf import OmegaConf
from io import StringIO
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

from retry import retry

# these are for testing txt2vid, should be removed and we should use things from our own code. 
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

#will be used for saving and reading a video made by the txt2vid function
import imageio, io

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
if (os.path.exists("configs/webui/userconfig_streamlit.yaml")):
	user_defaults = OmegaConf.load("configs/webui/userconfig_streamlit.yaml");
	defaults = OmegaConf.merge(defaults, user_defaults)

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

# should and will be moved to a settings menu in the UI at some point
save_format = [s.lower() for s in defaults.general.save_format.split(':')]
save_lossless = False
save_quality = 100
if save_format[0] == 'png':
	save_ext = 'png'
	save_format = 'png'
elif save_format[0] in ['jpg', 'jpeg']:
	save_quality = int(save_format[1]) if len(save_format) > 1 else 100
	save_ext = 'jpg'
	save_format = 'jpeg'
elif save_format[0] == 'webp':
	save_quality = int(save_format[1]) if len(save_format) > 1 else 100
	save_ext = 'webp'
	save_format = 'webp'
	if save_quality < 0: # e.g. webp:-100 for lossless mode
		save_lossless = True
		save_quality = abs(save_quality)

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(defaults.general.gpu)

@retry(tries=5)
def load_models(continue_prev_run = False, use_GFPGAN=False, use_RealESRGAN=False, RealESRGAN_model="RealESRGAN_x4plus",
                CustomModel_available=False, custom_model="Stable Diffusion v1.4"):
	"""Load the different models. We also reuse the models that are already in memory to speed things up instead of loading them again. """

	print ("Loading models.")

	st.session_state["progress_bar_text"].text("Loading models...")

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
		if "model" in st.session_state and st.session_state["custom_model"] == custom_model:
			print("Model already loaded")
		else:
			try:
				del st.session_state["model"]
			except KeyError:
				pass
			
			config = OmegaConf.load(defaults.general.default_model_config)
			
			if custom_model == defaults.general.default_model:
				model = load_model_from_config(config, defaults.general.default_model_path)			
			else:
				model = load_model_from_config(config, os.path.join("models","custom", f"{custom_model}.ckpt")) 
				
			st.session_state["custom_model"] = custom_model
			st.session_state["device"] = torch.device(f"cuda:{defaults.general.gpu}") if torch.cuda.is_available() else torch.device("cpu")
			st.session_state["model"] = (model if defaults.general.no_half else model.half()).to(st.session_state["device"] ) 			
	else:
		config = OmegaConf.load(defaults.general.default_model_config)

		if custom_model == defaults.general.default_model:
			model = load_model_from_config(config, defaults.general.default_model_path)			
		else:
			model = load_model_from_config(config, os.path.join("models","custom", f"{custom_model}.ckpt")) 
		
		st.session_state["custom_model"] = custom_model		
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
		if st.session_state["generation_mode"] == "img2img":
			round_sampling_steps = round(st.session_state.sampling_steps * st.session_state["denoising_strength"])
			percent = int(100 * float(i+1 if i+1 < round_sampling_steps else round_sampling_steps)/float(round_sampling_steps))
			st.session_state["progress_bar_text"].text(
				f"""Running step: {i+1 if i+1 < round_sampling_steps else round_sampling_steps}/{round_sampling_steps} {percent if percent < 100 else 100}%""")
		else:
			if st.session_state["generation_mode"] == "txt2vid":
				percent = int(100 * float(i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps)/float(st.session_state.sampling_steps))
				st.session_state["progress_bar_text"].text(
				        f"Running step: {i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps}"
				        f"{percent if percent < 100 else 100}%")	

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


def optimize_update_preview_frequency(current_chunk_speed, previous_chunk_speed, update_preview_frequency):
	"""Find the optimal update_preview_frequency value maximizing 
	performance while minimizing the time between updates."""
	if current_chunk_speed >= previous_chunk_speed:
		#print(f"{current_chunk_speed} >= {previous_chunk_speed}")
		update_preview_frequency +=1
		previous_chunk_speed = current_chunk_speed
	else:
		#print(f"{current_chunk_speed} <= {previous_chunk_speed}")
		update_preview_frequency -=1
		previous_chunk_speed = current_chunk_speed
		
	return current_chunk_speed, previous_chunk_speed, update_preview_frequency

# -----------------------------------------------------------------------------

@torch.no_grad()
def diffuse(
        pipe,
        cond_embeddings, # text conditioning, should be (1, 77, 768)
        cond_latents,    # image conditioning, should be (1, 4, 64, 64)
        num_inference_steps,
        cfg_scale,
        eta,
        ):
	
	torch_device = cond_latents.get_device()

	# classifier guidance: add the unconditional embedding
	max_length = cond_embeddings.shape[1] # 77
	uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
	uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]
	text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

	# if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
	if isinstance(pipe.scheduler, LMSDiscreteScheduler):
		cond_latents = cond_latents * pipe.scheduler.sigmas[0]

	# init the scheduler
	accepts_offset = "offset" in set(inspect.signature(pipe.scheduler.set_timesteps).parameters.keys())
	extra_set_kwargs = {}
	if accepts_offset:
		extra_set_kwargs["offset"] = 1
		
	pipe.scheduler.set_timesteps(num_inference_steps + st.session_state.sampling_steps, **extra_set_kwargs)
	# prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
	# eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
	# eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
	# and should be between [0, 1]
	accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
	extra_step_kwargs = {}
	if accepts_eta:
		extra_step_kwargs["eta"] = eta

	
	step_counter = 0
	inference_counter = 0
	current_chunk_speed = 0
	previous_chunk_speed = 0
	
	# diffuse!
	for i, t in enumerate(pipe.scheduler.timesteps):
		start = timeit.default_timer()
	
		#status_text.text(f"Running step: {step_counter}{total_number_steps} {percent} | {duration:.2f}{speed}")		

		# expand the latents for classifier free guidance
		latent_model_input = torch.cat([cond_latents] * 2)
		if isinstance(pipe.scheduler, LMSDiscreteScheduler):
			sigma = pipe.scheduler.sigmas[i]
			latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

		# predict the noise residual
		noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

		# cfg
		noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
		noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

		# compute the previous noisy sample x_t -> x_t-1
		if isinstance(pipe.scheduler, LMSDiscreteScheduler):
			cond_latents = pipe.scheduler.step(noise_pred, i, cond_latents, **extra_step_kwargs)["prev_sample"]
		else:
			cond_latents = pipe.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]
		
		#print (st.session_state["update_preview_frequency"])
		#update the preview image if it is enabled and the frequency matches the step_counter
		if defaults.general.update_preview:
			step_counter += 1
			
			if st.session_state.dynamic_preview_frequency:
				current_chunk_speed, previous_chunk_speed, defaults.general.update_preview_frequency = optimize_update_preview_frequency(
				        current_chunk_speed, previous_chunk_speed, defaults.general.update_preview_frequency)   
			
			if defaults.general.update_preview_frequency == step_counter or step_counter == st.session_state.sampling_steps:
				#scale and decode the image latents with vae
				cond_latents_2 = 1 / 0.18215 * cond_latents
				image_2 = pipe.vae.decode(cond_latents_2)
				
				# generate output numpy image as uint8
				image_2 = (image_2 / 2 + 0.5).clamp(0, 1)
				image_2 = image_2.cpu().permute(0, 2, 3, 1).numpy()
				image_2 = (image_2[0] * 255).astype(np.uint8)		
				
				st.session_state["preview_image"].image(image_2)
				
				step_counter = 0
		
		duration = timeit.default_timer() - start
		
		current_chunk_speed = duration
	
		if duration >= 1:
			speed = "s/it"
		else:
			speed = "it/s"
			duration = 1 / duration	
			
		if i > st.session_state.sampling_steps:
			inference_counter += 1
			inference_percent = int(100 * float(inference_counter if inference_counter < num_inference_steps else num_inference_steps)/float(num_inference_steps))
			inference_progress = f"{inference_counter if inference_counter < num_inference_steps else num_inference_steps}/{num_inference_steps} {inference_percent}% "
		else:
			inference_progress = ""
				
		percent = int(100 * float(i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps)/float(st.session_state.sampling_steps))
		frames_percent = int(100 * float(st.session_state.current_frame if st.session_state.current_frame < st.session_state.max_frames else st.session_state.max_frames)/float(st.session_state.max_frames))
		
		st.session_state["progress_bar_text"].text(
	                f"Running step: {i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps} "
	                f"{percent if percent < 100 else 100}% {inference_progress}{duration:.2f}{speed} | "
		        f"Frame: {st.session_state.current_frame if st.session_state.current_frame < st.session_state.max_frames else st.session_state.max_frames}/{st.session_state.max_frames} "
		        f"{frames_percent if frames_percent < 100 else 100}% {st.session_state.frame_duration:.2f}{st.session_state.frame_speed}"
		)
		st.session_state["progress_bar"].progress(percent if percent < 100 else 100)		

	# scale and decode the image latents with vae
	cond_latents = 1 / 0.18215 * cond_latents
	image = pipe.vae.decode(cond_latents)

	# generate output numpy image as uint8
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.cpu().permute(0, 2, 3, 1).numpy()
	image = (image[0] * 255).astype(np.uint8)

	return image


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
	
	if type(s) is list:
		seed_list = []
		for seed in s:
			if seed is None or seed == '':
				seed_list.append(random.randint(0, 2**32 - 1))
			else:
				seed_list = s
				
		return seed_list
	
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

	if defaults.general.save_metadata or write_info_files:
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
		metadata = \
			dict(
				target="txt2img" if init_img is None else "img2img",
				prompt=prompts[i], ddim_steps=steps, toggles=toggles, sampler_name=sampler_name,
				ddim_eta=ddim_eta, n_iter=n_iter, batch_size=batch_size, cfg_scale=cfg_scale,
				seed=seeds[i], width=width, height=height, normalize_prompt_weights=normalize_prompt_weights)
		# Not yet any use for these, but they bloat up the files:
		# info_dict["init_img"] = init_img
		# info_dict["init_mask"] = init_mask
		if init_img is not None:
			metadata["denoising_strength"] = str(denoising_strength)
			metadata["resize_mode"] = resize_mode

	if write_info_files:
		with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
			yaml.dump(metadata, f, allow_unicode=True, width=10000)

	if defaults.general.save_metadata:
		# metadata = {
		# 	"SD:prompt": prompts[i],
		# 	"SD:seed": str(seeds[i]),
		# 	"SD:width": str(width),
		# 	"SD:height": str(height),
		# 	"SD:steps": str(steps),
		# 	"SD:cfg_scale": str(cfg_scale),
		# 	"SD:normalize_prompt_weights": str(normalize_prompt_weights),
		# }
		metadata = {"SD:" + k:v for (k,v) in metadata.items()}

		if save_ext == "png":
			mdata = PngInfo()
			for key in metadata:
				mdata.add_text(key, str(metadata[key]))
			image.save(f"{filename_i}.png", pnginfo=mdata)
		else:
			if jpg_sample:
				image.save(f"{filename_i}.jpg", quality=save_quality,
						   optimize=True)
			elif save_ext == "webp":
				image.save(f"{filename_i}.{save_ext}", f"webp", quality=save_quality,
						   lossless=save_lossless)
			else:
				# not sure what file format this is
				image.save(f"{filename_i}.{save_ext}", f"{save_ext}")
			try:
				exif_dict = piexif.load(f"{filename_i}.{save_ext}")
			except:
				exif_dict = { "Exif": dict() }
			exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
				json.dumps(metadata), encoding="unicode")
			piexif.insert(piexif.dump(exif_dict), f"{filename_i}.{save_ext}")

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


import find_noise_for_image
import matched_noise


def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, save_grid, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        fp=None, ddim_eta=0.0, normalize_prompt_weights=True, init_img=None, init_mask=None,
        mask_blur_strength=3, mask_restore=False, denoising_strength=0.75, noise_mode=0, find_noise_steps=1, resize_mode=None, uses_loopback=False,
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

			if noise_mode == 1 or noise_mode == 3:
				# TODO params for find_noise_to_image
				x = torch.cat(batch_size * [find_noise_for_image.find_noise_for_image(
					st.session_state["model"], st.session_state["device"],
					init_img.convert('RGB'), '', find_noise_steps, 0.0, normalize=True,
					generation_callback=generation_callback,
				)], dim=0)
			else:
				# we manually generate all input noises because each one should have a specific seed
				x = create_random_tensors(shape, seeds=seeds)

			if variant_amount > 0.0: # we are making variants
				# using variant_seed as sneaky toggle, 
				# when not None or '' use the variant_seed
				# otherwise use seeds
				if variant_seed != None and variant_seed != '':
					specified_variant_seed = seed_to_int(variant_seed)
					torch.manual_seed(specified_variant_seed)
					seeds = [specified_variant_seed]
				# finally, slerp base_x noise to target_x noise for creating a variant
				x = slerp(defaults.general.gpu, max(0.0, min(1.0, variant_amount)), base_x, x)

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
					   
				if mask_restore and init_mask:
					#init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
					init_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
					init_mask = init_mask.convert('L')
					init_img = init_img.convert('RGB')
					image = image.convert('RGB')

					if use_RealESRGAN and st.session_state["RealESRGAN"] is not None:
						if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
							#try_loading_RealESRGAN(realesrgan_model_name)
							load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

						output, img_mode = st.session_state["RealESRGAN"].enhance(np.array(init_img, dtype=np.uint8))
						init_img = Image.fromarray(output)
						init_img = init_img.convert('RGB')

						output, img_mode = st.session_state["RealESRGAN"].enhance(np.array(init_mask, dtype=np.uint8))
						init_mask = Image.fromarray(output)
						init_mask = init_mask.convert('L')

					image = Image.composite(init_img, image, init_mask)
						
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

import skimage

def img2img(prompt: str = '', init_info: any = None, init_info_mask: any = None, mask_mode: int = 0, mask_blur_strength: int = 3, 
            mask_restore: bool = False, ddim_steps: int = 50, sampler_name: str = 'DDIM',
            n_iter: int = 1,  cfg_scale: float = 7.5, denoising_strength: float = 0.8,
            seed: int = -1, noise_mode: int = 0, find_noise_steps: str = "", height: int = 512, width: int = 512, resize_mode: int = 0, fp = None,
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

	def process_init_mask(init_mask: Image):
		if init_mask.mode == "RGBA":
			init_mask = init_mask.convert('RGBA')
			background = Image.new('RGBA', init_mask.size, (0, 0, 0))
			init_mask = Image.alpha_composite(background, init_mask)
			init_mask = init_mask.convert('RGB')
		return init_mask

	init_img = init_info
	init_mask = None
	if mask_mode == 0:
		if init_info_mask:
			init_mask = process_init_mask(init_info_mask)
	elif mask_mode == 1:
		if init_info_mask:
			init_mask = process_init_mask(init_info_mask)
			init_mask = ImageOps.invert(init_mask)
	elif mask_mode == 2:
		init_img_transparency = init_img.split()[-1].convert('L')#.point(lambda x: 255 if x > 0 else 0, mode='1')
		init_mask = init_img_transparency
		init_mask = init_mask.convert("RGB")
		init_mask = resize_image(resize_mode, init_mask, width, height)
		init_mask = init_mask.convert("RGB")

	assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
	t_enc = int(denoising_strength * ddim_steps)

	if init_mask is not None and (noise_mode == 2 or noise_mode == 3) and init_img is not None:
		noise_q = 0.99
		color_variation = 0.0
		mask_blend_factor = 1.0

		np_init = (np.asarray(init_img.convert("RGB"))/255.0).astype(np.float64) # annoyingly complex mask fixing
		np_mask_rgb = 1. - (np.asarray(ImageOps.invert(init_mask).convert("RGB"))/255.0).astype(np.float64)
		np_mask_rgb -= np.min(np_mask_rgb)
		np_mask_rgb /= np.max(np_mask_rgb)
		np_mask_rgb = 1. - np_mask_rgb
		np_mask_rgb_hardened = 1. - (np_mask_rgb < 0.99).astype(np.float64)
		blurred = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., channel_axis=2, truncate=32.)
		blurred2 = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., channel_axis=2, truncate=32.)
		#np_mask_rgb_dilated = np_mask_rgb + blurred  # fixup mask todo: derive magic constants
		#np_mask_rgb = np_mask_rgb + blurred
		np_mask_rgb_dilated = np.clip((np_mask_rgb + blurred2) * 0.7071, 0., 1.)
		np_mask_rgb = np.clip((np_mask_rgb + blurred) * 0.7071, 0., 1.)

		noise_rgb = matched_noise.get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
		blend_mask_rgb = np.clip(np_mask_rgb_dilated,0.,1.) ** (mask_blend_factor)
		noised = noise_rgb[:]
		blend_mask_rgb **= (2.)
		noised = np_init[:] * (1. - blend_mask_rgb) + noised * blend_mask_rgb

		np_mask_grey = np.sum(np_mask_rgb, axis=2)/3.
		ref_mask = np_mask_grey < 1e-3
		
		all_mask = np.ones((height, width), dtype=bool)
		noised[all_mask,:] = skimage.exposure.match_histograms(noised[all_mask,:]**1., noised[ref_mask,:], channel_axis=1)
		
		init_img = Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")
		st.session_state["editor_image"].image(init_img) # debug

	def init():
		image = init_img.convert('RGB')
		image = np.array(image).astype(np.float32) / 255.0
		image = image[None].transpose(0, 3, 1, 2)
		image = torch.from_numpy(image)

		mask_channel = None
		if init_mask:
			alpha = resize_image(resize_mode, init_mask, width // 8, height // 8)
			mask_channel = alpha.split()[-1]

		mask = None
		if mask_channel is not None:
			mask = np.array(mask_channel).astype(np.float32) / 255.0
			mask = (1 - mask)
			mask = np.tile(mask, (4, 1, 1))
			mask = mask[None].transpose(0, 1, 2, 3)
			mask = torch.from_numpy(mask).to(st.session_state["device"])
		
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

		for i in range(n_iter):
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
                                n_iter=1,
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
                                mask_blur_strength=mask_blur_strength,
                                mask_restore=mask_restore,
                                denoising_strength=denoising_strength,
                                noise_mode=noise_mode,
                                find_noise_steps=find_noise_steps,
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
                        mask_blur_strength=mask_blur_strength,
                        denoising_strength=denoising_strength,
                        noise_mode=noise_mode,
                        find_noise_steps=find_noise_steps,
                        mask_restore=mask_restore,
                        resize_mode=resize_mode,
                        uses_loopback=loopback,
                        sort_samples=group_by_prompt,
                        write_info_files=write_info_files,
                        jpg_sample=save_as_jpg
                )

	del sampler

	return output_images, seed, info, stats

@retry((RuntimeError, KeyError) , tries=3)
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


#
def txt2vid(
                # --------------------------------------
                # args you probably want to change
                prompts = ["blueberry spaghetti", "strawberry spaghetti"], # prompt to dream about
                gpu:int = defaults.general.gpu, # id of the gpu to run on
                #name:str = 'test', # name of this project, for the output directory
                #rootdir:str = defaults.general.outdir,
                num_steps:int = 200, # number of steps between each pair of sampled points
                max_frames:int = 10000, # number of frames to write and then exit the script
                num_inference_steps:int = 50, # more (e.g. 100, 200 etc) can create slightly better images
                cfg_scale:float = 5.0, # can depend on the prompt. usually somewhere between 3-10 is good
                do_loop = False,
                use_lerp_for_text = False,
                seeds = None,
                quality:int = 100, # for jpeg compression of the output images
                eta:float = 0.0,
                width:int = 256,
                height:int = 256,
                weights_path = "CompVis/stable-diffusion-v1-4",
                scheduler="klms",  # choices: default, ddim, klms
                disable_tqdm = False,
                #-----------------------------------------------
                beta_start = 0.0001,
                beta_end = 0.00012,
                beta_schedule = "scaled_linear"
                ):	
	"""
	prompt = ["blueberry spaghetti", "strawberry spaghetti"], # prompt to dream about
	gpu:int = defaults.general.gpu, # id of the gpu to run on
	#name:str = 'test', # name of this project, for the output directory
	#rootdir:str = defaults.general.outdir,
	num_steps:int = 200, # number of steps between each pair of sampled points
	max_frames:int = 10000, # number of frames to write and then exit the script
	num_inference_steps:int = 50, # more (e.g. 100, 200 etc) can create slightly better images
	cfg_scale:float = 5.0, # can depend on the prompt. usually somewhere between 3-10 is good
	do_loop = False,
	use_lerp_for_text = False,
	seed = None,
	quality:int = 100, # for jpeg compression of the output images
	eta:float = 0.0,
	width:int = 256,
	height:int = 256,
	weights_path = "CompVis/stable-diffusion-v1-4",
	scheduler="klms",  # choices: default, ddim, klms
	disable_tqdm = False,
	beta_start = 0.0001,
	beta_end = 0.00012,
	beta_schedule = "scaled_linear"
	"""
	mem_mon = MemUsageMonitor('MemMon')
	mem_mon.start()	
	
	
	seeds = seed_to_int(seeds)
	
	# We add an extra frame because most
	# of the time the first frame is just the noise.
	max_frames +=1
	
	assert torch.cuda.is_available()
	assert height % 8 == 0 and width % 8 == 0
	torch.manual_seed(seeds)
	torch_device = f"cuda:{gpu}"
	
	# init the output dir
	sanitized_prompt = slugify(prompts)
	
	full_path = os.path.join(os.getcwd(), defaults.general.outdir, "txt2vid-samples", "samples", sanitized_prompt)
	
	if len(full_path) > 220:
		sanitized_prompt = sanitized_prompt[:220-len(full_path)]
		full_path = os.path.join(os.getcwd(), defaults.general.outdir, "txt2vid-samples", "samples", sanitized_prompt)
		
	os.makedirs(full_path, exist_ok=True)
	
	# Write prompt info to file in output dir so we can keep track of what we did
	if st.session_state.write_info_files:
		with open(os.path.join(full_path , f'{slugify(str(seeds))}_config.json' if len(prompts) > 1 else "prompts_config.json"), "w") as outfile:
			outfile.write(json.dumps(
				dict(
				prompts = prompts,
				gpu = gpu,
				num_steps = num_steps,
				max_frames = max_frames,
				num_inference_steps = num_inference_steps,
				cfg_scale = cfg_scale,
				do_loop = do_loop,
				use_lerp_for_text = use_lerp_for_text,
				seeds = seeds,
				quality = quality,
				eta = eta,
				width = width,
				height = height,
				weights_path = weights_path,
				scheduler=scheduler,
				disable_tqdm = disable_tqdm,
				beta_start = beta_start,
				beta_end = beta_end,
				beta_schedule = beta_schedule
				                ),
			    indent=2,
			    sort_keys=False,
			    ))

	#print(scheduler)
	default_scheduler = PNDMScheduler(
		beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule
	)
	# ------------------------------------------------------------------------------
	#Schedulers
	ddim_scheduler = DDIMScheduler(
		beta_start=beta_start,
	    beta_end=beta_end,
	    beta_schedule=beta_schedule,
	    clip_sample=False,
	    set_alpha_to_one=False,
	)
	
	klms_scheduler = LMSDiscreteScheduler(
		beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule
	)
		
	SCHEDULERS = dict(default=default_scheduler, ddim=ddim_scheduler, klms=klms_scheduler)		
	
	# ------------------------------------------------------------------------------
	
	#if weights_path == "Stable Diffusion v1.4":
		#weights_path = "CompVis/stable-diffusion-v1-4"
	#else:
		#weights_path = os.path.join("./models", "custom", f"{weights_path}.ckpt")
	
	try:
		if "model" in st.session_state:
			del st.session_state["model"]
	except:
		pass
	
	#print (st.session_state["weights_path"] != weights_path)
	
	try:
		if not st.session_state["pipe"] or st.session_state["weights_path"] != weights_path:
			if st.session_state["weights_path"] != weights_path:
				del st.session_state["weights_path"]
			
			st.session_state["weights_path"] = weights_path	
			st.session_state["pipe"] = StableDiffusionPipeline.from_pretrained(
				weights_path,
				use_local_file=True,
				use_auth_token=True,
				#torch_dtype=torch.float16 if not defaults.general.no_half else None,
				revision="fp16" if not defaults.general.no_half else None
			)
		
			st.session_state["pipe"].unet.to(torch_device)
			st.session_state["pipe"].vae.to(torch_device)
			st.session_state["pipe"].text_encoder.to(torch_device)
			print("Tx2Vid Model Loaded")
		else:
			print("Tx2Vid Model already Loaded")
		
	except:
		#del st.session_state["weights_path"]
		#del st.session_state["pipe"]
		
		st.session_state["weights_path"] = weights_path
		st.session_state["pipe"] = StableDiffusionPipeline.from_pretrained(
			                weights_path,
			                use_local_file=True,
			                use_auth_token=True,
			                #torch_dtype=torch.float16 if not defaults.general.no_half else None,
			                revision="fp16" if not defaults.general.no_half else None
			        )
	
		st.session_state["pipe"].unet.to(torch_device)
		st.session_state["pipe"].vae.to(torch_device)
		st.session_state["pipe"].text_encoder.to(torch_device)
		print("Tx2Vid Model Loaded")		

	st.session_state["pipe"].scheduler = SCHEDULERS[scheduler]
	
	# get the conditional text embeddings based on the prompt
	text_input = st.session_state["pipe"].tokenizer(prompts, padding="max_length", max_length=st.session_state["pipe"].tokenizer.model_max_length, truncation=True, return_tensors="pt")
	cond_embeddings = st.session_state["pipe"].text_encoder(text_input.input_ids.to(torch_device))[0] # shape [1, 77, 768]

	# sample a source
	init1 = torch.randn((1, st.session_state["pipe"].unet.in_channels, height // 8, width // 8), device=torch_device)

	if do_loop:
		prompts = [prompts, prompts]
		seeds = [seeds, seeds]
		#first_seed, *seeds = seeds
		#prompts.append(prompts)
		#seeds.append(first_seed)
	
	
	# iterate the loop
	frames = []
	frame_index = 0
	
	st.session_state["frame_total_duration"] = 0
	st.session_state["frame_total_speed"] = 0	
	
	try:
		while frame_index < max_frames:
			st.session_state["frame_duration"] = 0
			st.session_state["frame_speed"] = 0			
			st.session_state["current_frame"] = frame_index

			# sample the destination
			init2 = torch.randn((1, st.session_state["pipe"].unet.in_channels, height // 8, width // 8), device=torch_device)

			for i, t in enumerate(np.linspace(0, 1, num_steps)):
				start = timeit.default_timer()
				print(f"COUNT: {frame_index+1}/{num_steps}")
			
				#if use_lerp_for_text:
					#init = torch.lerp(init1, init2, float(t))
				#else:
					#init = slerp(gpu, float(t), init1, init2)
				
				init = slerp(gpu, float(t), init1, init2)
				
				with autocast("cuda"):
					image = diffuse(st.session_state["pipe"], cond_embeddings, init, num_inference_steps, cfg_scale, eta)

				im = Image.fromarray(image)
				outpath = os.path.join(full_path, 'frame%06d.png' % frame_index)
				im.save(outpath, quality=quality)

				# send the image to the UI to update it
				#st.session_state["preview_image"].image(im) 	

				#append the frames to the frames list so we can use them later.
				frames.append(np.asarray(im))

				#increase frame_index counter.
				frame_index += 1

				st.session_state["current_frame"] = frame_index

				duration = timeit.default_timer() - start

				if duration >= 1:
					speed = "s/it"
				else:
					speed = "it/s"
					duration = 1 / duration	

				st.session_state["frame_duration"] = duration
				st.session_state["frame_speed"] = speed

			init1 = init2

	except StopException:
		pass
		
		
	if st.session_state['save_video']:
		# write video to memory
		#output = io.BytesIO()
		#writer = imageio.get_writer(os.path.join(os.getcwd(), defaults.general.outdir, "txt2vid-samples"), im, extension=".mp4", fps=30)
		try:
			video_path = os.path.join(os.getcwd(), defaults.general.outdir, "txt2vid-samples","temp.mp4")
			writer = imageio.get_writer(video_path, fps=24)
			for frame in frames:
				writer.append_data(frame)
			writer.close()		
		except:
			print("Can't save video, skipping.")
		
		# show video preview on the UI
		st.session_state["preview_video"].video(open(video_path, 'rb').read())
	
	mem_max_used, mem_total = mem_mon.read_and_stop()
	time_diff = time.time()- start	
	
	info = f"""
                {prompts}
                Sampling Steps: {num_steps}, Sampler: {scheduler}, CFG scale: {cfg_scale}, Seed: {seeds}, Max Frames: {max_frames}""".strip()
	stats = f'''
                Took { round(time_diff, 2) }s total ({ round(time_diff/(max_frames),2) }s per image)
                Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

	return im, seeds, info, stats


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

	st.set_page_config(page_title="Stable Diffusion Playground", layout="wide")

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
		
	# Allow for custom models to be used instead of the default one,
	# an example would be Waifu-Diffusion or any other fine tune of stable diffusion
	custom_models:sorted = []
	for root, dirs, files in os.walk(os.path.join("models", "custom")):
		for file in files:
			if os.path.splitext(file)[1] == '.ckpt':
				fullpath = os.path.join(root, file)
				#print(fullpath)
				custom_models.append(os.path.splitext(file)[0])
				#print (os.path.splitext(file)[0])
	
	if len(custom_models) > 0:
		CustomModel_available = True
		custom_models.append("Stable Diffusion v1.4")
	else:
		CustomModel_available = False

	with st.sidebar:
		# The global settings section will be moved to the Settings page.
		#with st.expander("Global Settings:"):
		#st.write("Global Settings:")
		#defaults.general.update_preview = st.checkbox("Update Image Preview", value=defaults.general.update_preview,
                                                              #help="If enabled the image preview will be updated during the generation instead of at the end. You can use the Update Preview \
							      #Frequency option bellow to customize how frequent it's updated. By default this is enabled and the frequency is set to 1 step.")
		#st.session_state.update_preview_frequency = st.text_input("Update Image Preview Frequency", value=defaults.general.update_preview_frequency,
                                                                          #help="Frequency in steps at which the the preview image is updated. By default the frequency is set to 1 step.")
		
		tabs = on_hover_tabs(tabName=['Stable Diffusion', "Textual Inversion","Model Manager","Settings"], 
                         iconName=['dashboard','model_training' ,'cloud_download', 'settings'], default_choice=0)
		
	
	if tabs =='Stable Diffusion':		
		txt2img_tab, img2img_tab, txt2vid_tab, postprocessing_tab = st.tabs(["Text-to-Image Unified", "Image-to-Image Unified", 
			                                                                                "Text-to-Video","Post-Processing"])		
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
						
					with st.expander("Preview Settings"):
						st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=defaults.txt2img.update_preview,
					                                                         help="If enabled the image preview will be updated during the generation instead of at the end. \
					                                                         You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
					                                                         By default this is enabled and the frequency is set to 1 step.")
						
						st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=defaults.txt2img.update_preview_frequency,
					                                                                  help="Frequency in steps at which the the preview image is updated. By default the frequency \
					                                                                  is set to 1 step.")						
	
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
						st.session_state["preview_video"] = st.empty()
	
						st.session_state["loading"] = st.empty()
	
						st.session_state["progress_bar_text"] = st.empty()
						st.session_state["progress_bar"] = st.empty()
	
						message = st.empty()
	
					with gallery_tab:
						st.write('Here should be the image gallery, if I could make a grid in streamlit.')
	
				with col3:
					# If we have custom models available on the "models/custom" 
					#folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
					if CustomModel_available:
						custom_model = st.selectbox("Custom Model:", custom_models,
							    index=custom_models.index(defaults.general.default_model),
							    help="Select the model you want to use. This option is only available if you have custom models \
							    on your 'models/custom' folder. The model name that will be shown here is the same as the name\
							    the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
							    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4") 	
					else:
						custom_model = "Stable Diffusion v1.4"
					
					st.session_state.sampling_steps = st.slider("Sampling Steps", value=defaults.txt2img.sampling_steps, min_value=1, max_value=250)
					
					sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
					sampler_name = st.selectbox("Sampling method", sampler_name_list,
						    index=sampler_name_list.index(defaults.txt2img.default_sampler), help="Sampling method to use. Default: k_euler")  
	
	
	
					#basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])
	
					#with basic_tab:
						#summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
							#help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")
	
					with st.expander("Advanced"):
						separate_prompts = st.checkbox("Create Prompt Matrix.", value=False,
						                               help="Separate multiple prompts using the `|` character, and get all combinations of them.")
						normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.",
						                                       value=defaults.txt2img.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
						save_individual_images = st.checkbox("Save individual images.", value=defaults.txt2img.save_individual_images,
						                                     help="Save each image generated before any filter or enhancement is applied.")
						save_grid = st.checkbox("Save grid",value=defaults.txt2img.save_grid, help="Save a grid with all the images generated into a single image.")
						group_by_prompt = st.checkbox("Group results by prompt", value=defaults.txt2img.group_by_prompt,
							                      help="Saves all the images with the same prompt into the same folder. \
						                              When using a prompt matrix each prompt combination will have its own folder.")
						write_info_files = st.checkbox("Write Info file", value=defaults.txt2img.write_info_files,
						                               help="Save a file next to the image with informartion about the generation.")						
						save_as_jpg = st.checkbox("Save samples as jpg", value=defaults.txt2img.save_as_jpg, help="Saves the images as jpg instead of png.")
	
						if GFPGAN_available:
							use_GFPGAN = st.checkbox("Use GFPGAN", value=defaults.txt2img.use_GFPGAN,
							                         help="Uses the GFPGAN model to improve faces after the generation. This greatly improve the quality and \
							                         consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
						else:
							use_GFPGAN = False
	
						if RealESRGAN_available:
							use_RealESRGAN = st.checkbox("Use RealESRGAN", value=defaults.txt2img.use_RealESRGAN,
							                             help="Uses the RealESRGAN model to upscale the images after the generation. This greatly improve the \
							                             quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
							RealESRGAN_model = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
						else:
							use_RealESRGAN = False
							RealESRGAN_model = "RealESRGAN_x4plus"
	
						variant_amount = st.slider("Variant Amount:", value=defaults.txt2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
						variant_seed = st.text_input("Variant Seed:", value=defaults.txt2img.seed,
						                             help="The seed to use when generating a variant, if left blank a random seed will be generated.")
	
	
				if generate_button:
					#print("Loading models")
					# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.		
					load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model, CustomModel_available, custom_model)                
	
					try:
						output_images, seed, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, RealESRGAN_model, batch_count, 1, 
							                                   cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
							                                   save_grid, group_by_prompt, save_as_jpg, use_GFPGAN, use_RealESRGAN, RealESRGAN_model, fp=defaults.general.fp,
							                                   variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files)
					
						message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")
	
					except KeyError:
						output_images, seed, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, RealESRGAN_model, batch_count, 1, 
							                                   cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
							                                   save_grid, group_by_prompt, save_as_jpg, use_GFPGAN, use_RealESRGAN, RealESRGAN_model, fp=defaults.general.fp,
							                                   variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files)
					
						message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")
						
					except (StopException):
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
					# If we have custom models available on the "models/custom" 
					#folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
					if CustomModel_available:
						custom_model = st.selectbox("Custom Model:", custom_models,
							    index=custom_models.index(defaults.general.default_model),
							    help="Select the model you want to use. This option is only available if you have custom models \
							    on your 'models/custom' folder. The model name that will be shown here is the same as the name\
							    the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
							    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4") 	
					else:
						custom_model = "Stable Diffusion v1.4"
						
					st.session_state["sampling_steps"] = st.slider("Sampling Steps", value=defaults.img2img.sampling_steps, min_value=1, max_value=500)
					st.session_state["sampler_name"] = st.selectbox("Sampling method",
					                                                ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"],
					                                                index=sampler_name_list.index(defaults.img2img.sampler_name),
					                                                help="Sampling method to use.")
	
					mask_mode_list = ["Mask", "Inverted mask", "Image alpha"]
					mask_mode = st.selectbox("Mask Mode", mask_mode_list,
					                         help="Select how you want your image to be masked.\"Mask\" modifies the image where the mask is white.\n\
					                         \"Inverted mask\" modifies the image where the mask is black. \"Image alpha\" modifies the image where the image is transparent."
					)
					mask_mode = mask_mode_list.index(mask_mode)

					width = st.slider("Width:", min_value=64, max_value=1024, value=defaults.img2img.width, step=64)
					height = st.slider("Height:", min_value=64, max_value=1024, value=defaults.img2img.height, step=64)
					seed = st.text_input("Seed:", value=defaults.img2img.seed, help=" The seed to use, if left blank a random seed will be generated.")
					noise_mode_list = ["Seed", "Find Noise", "Matched Noise", "Find+Matched Noise"]
					noise_mode = st.selectbox(
						"Noise Mode", noise_mode_list,
						help=""
					)
					noise_mode = noise_mode_list.index(noise_mode)
					find_noise_steps = st.slider("Find Noise Steps", value=100, min_value=1, max_value=500)
					batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.img2img.batch_count, step=1,
					                        help="How many iterations or batches of images to generate in total.")
	
					#			
					with st.expander("Advanced"):
						separate_prompts = st.checkbox("Create Prompt Matrix.", value=defaults.img2img.separate_prompts,
						                               help="Separate multiple prompts using the `|` character, and get all combinations of them.")
						normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=defaults.img2img.normalize_prompt_weights,
						                                       help="Ensure the sum of all weights add up to 1.0")
						loopback = st.checkbox("Loopback.", value=defaults.img2img.loopback, help="Use images from previous batch when creating next batch.")
						random_seed_loopback = st.checkbox("Random loopback seed.", value=defaults.img2img.random_seed_loopback, help="Random loopback seed")
						save_individual_images = st.checkbox("Save individual images.", value=defaults.img2img.save_individual_images,
						                                     help="Save each image generated before any filter or enhancement is applied.")
						save_grid = st.checkbox("Save grid",value=defaults.img2img.save_grid, help="Save a grid with all the images generated into a single image.")
						group_by_prompt = st.checkbox("Group results by prompt", value=defaults.img2img.group_by_prompt,
							                      help="Saves all the images with the same prompt into the same folder. \
						                              When using a prompt matrix each prompt combination will have its own folder.")
						write_info_files = st.checkbox("Write Info file", value=defaults.img2img.write_info_files, 
						                               help="Save a file next to the image with informartion about the generation.")						
						save_as_jpg = st.checkbox("Save samples as jpg", value=defaults.img2img.save_as_jpg, help="Saves the images as jpg instead of png.")
	
						if GFPGAN_available:
							use_GFPGAN = st.checkbox("Use GFPGAN", value=defaults.img2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation.\
							This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
						else:
							use_GFPGAN = False
	
						if RealESRGAN_available:
							use_RealESRGAN = st.checkbox("Use RealESRGAN", value=defaults.img2img.use_RealESRGAN,
							                             help="Uses the RealESRGAN model to upscale the images after the generation.\
							This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
							RealESRGAN_model = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
						else:
							use_RealESRGAN = False
							RealESRGAN_model = "RealESRGAN_x4plus"
	
						variant_amount = st.slider("Variant Amount:", value=defaults.img2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
						variant_seed = st.text_input("Variant Seed:", value=defaults.img2img.variant_seed,
						                             help="The seed to use when generating a variant, if left blank a random seed will be generated.")
						cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.img2img.cfg_scale, step=0.5,
						                      help="How strongly the image should follow the prompt.")
						batch_size = st.slider("Batch size", min_value=1, max_value=100, value=defaults.img2img.batch_size, step=1,
							               help="How many images are at once in a batch.\
							                       It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish \
						                               generation as more images are generated at once.\
							                       Default: 1")
	
						st.session_state["denoising_strength"] = st.slider("Denoising Strength:", value=defaults.img2img.denoising_strength, 
						                                                   min_value=0.01, max_value=1.0, step=0.01)
	
					with st.expander("Preview Settings"):
						st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=defaults.img2img.update_preview,
					                                                         help="If enabled the image preview will be updated during the generation instead of at the end. \
					                                                         You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
					                                                         By default this is enabled and the frequency is set to 1 step.")
						
						st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=defaults.img2img.update_preview_frequency,
					                                                                  help="Frequency in steps at which the the preview image is updated. By default the frequency \
					                                                                  is set to 1 step.")						
					
				with col2_img2img_layout:
					editor_tab = st.tabs(["Editor"])
	
					editor_image = st.empty()
					st.session_state["editor_image"] = editor_image
	
					refresh_button = st.form_submit_button("Refresh")

					masked_image_holder = st.empty()
					image_holder = st.empty()

					uploaded_images = st.file_uploader(
						"Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "webp"],
						help="Upload an image which will be used for the image to image generation.",
					)
					if uploaded_images:
						image = Image.open(uploaded_images).convert('RGBA')
						new_img = image.resize((width, height))
						image_holder.image(new_img)
					
					mask_holder = st.empty()

					uploaded_masks = st.file_uploader(
						"Upload Mask", accept_multiple_files=False, type=["png", "jpg", "jpeg", "webp"],
						help="Upload an mask image which will be used for masking the image to image generation.",
					)
					if uploaded_masks:
						mask = Image.open(uploaded_masks)
						if mask.mode == "RGBA":
							mask = mask.convert('RGBA')
							background = Image.new('RGBA', mask.size, (0, 0, 0))
							mask = Image.alpha_composite(background, mask)
						mask = mask.resize((width, height))
						mask_holder.image(mask)

					if uploaded_images and uploaded_masks:
						if mask_mode != 2:
							final_img = new_img.copy()
							alpha_layer = mask.convert('L')
							strength = st.session_state["denoising_strength"]
							if mask_mode == 0:
								alpha_layer = ImageOps.invert(alpha_layer)
								alpha_layer = alpha_layer.point(lambda a: a * strength)
								alpha_layer = ImageOps.invert(alpha_layer)
							elif mask_mode == 1:
								alpha_layer = alpha_layer.point(lambda a: a * strength)
								alpha_layer = ImageOps.invert(alpha_layer)
								
							final_img.putalpha(alpha_layer)
							
							with masked_image_holder.container():
								st.text("Masked Image Preview")
								st.image(final_img)
	
	
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
					load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model, CustomModel_available, custom_model)                
					if uploaded_images:
						image = Image.open(uploaded_images).convert('RGBA')
						new_img = image.resize((width, height))
						#img_array = np.array(image) # if you want to pass it to OpenCV
						new_mask = None
						if uploaded_masks:
							mask = Image.open(uploaded_masks).convert('RGBA')
							new_mask = mask.resize((width, height))
	
						try:
							output_images, seed, info, stats = img2img(prompt=prompt, init_info=new_img, init_info_mask=new_mask, mask_mode=mask_mode, ddim_steps=st.session_state["sampling_steps"],
								                                   sampler_name=st.session_state["sampler_name"], n_iter=batch_count,
								                                   cfg_scale=cfg_scale, denoising_strength=st.session_state["denoising_strength"], variant_seed=variant_seed,
								                                   seed=seed, noise_mode=noise_mode, find_noise_steps=find_noise_steps, width=width, height=height, fp=defaults.general.fp, variant_amount=variant_amount, 
								                                   ddim_eta=0.0, write_info_files=write_info_files, RealESRGAN_model=RealESRGAN_model,
								                                   separate_prompts=separate_prompts, normalize_prompt_weights=normalize_prompt_weights,
								                                   save_individual_images=save_individual_images, save_grid=save_grid, 
								                                   group_by_prompt=group_by_prompt, save_as_jpg=save_as_jpg, use_GFPGAN=use_GFPGAN,
								                                   use_RealESRGAN=use_RealESRGAN if not loopback else False, loopback=loopback
								                                   )
		
							#show a message when the generation is complete.
							message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")
	
						except (StopException, KeyError):
							print(f"Received Streamlit StopException")
	
					# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
					# use the current col2 first tab to show the preview_img and update it as its generated.
					#preview_image.image(output_images, width=750)
					
		with txt2vid_tab:
			with st.form("txt2vid-inputs"):
				st.session_state["generation_mode"] = "txt2vid"
		
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
					width = st.slider("Width:", min_value=64, max_value=2048, value=defaults.txt2vid.width, step=64)
					height = st.slider("Height:", min_value=64, max_value=2048, value=defaults.txt2vid.height, step=64)
					cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.txt2vid.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")
					seed = st.text_input("Seed:", value=defaults.txt2vid.seed, help=" The seed to use, if left blank a random seed will be generated.")
					batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.txt2vid.batch_count, step=1, help="How many iterations or batches of images to generate in total.")
					#batch_size = st.slider("Batch size", min_value=1, max_value=250, value=defaults.txt2vid.batch_size, step=1,
						#help="How many images are at once in a batch.\
						#It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
						#Default: 1")
						
					st.session_state["max_frames"] = int(st.text_input("Max Frames:", value=defaults.txt2vid.max_frames, help="Specify the max number of frames you want to generate."))
					
					with st.expander("Preview Settings"):
						st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=defaults.txt2vid.update_preview,
						                                                 help="If enabled the image preview will be updated during the generation instead of at the end. \
						                                                 You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
						                                                 By default this is enabled and the frequency is set to 1 step.")
						
						st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=defaults.txt2vid.update_preview_frequency,
												          help="Frequency in steps at which the the preview image is updated. By default the frequency \
						                                                          is set to 1 step.")						
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
						
						generate_video = st.empty()
						st.session_state["preview_video"] = st.empty()
		
						message = st.empty()
		
					with gallery_tab:
						st.write('Here should be the image gallery, if I could make a grid in streamlit.')
		
				with col3:
					# If we have custom models available on the "models/custom" 
					#folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
					#if CustomModel_available:
					custom_model = st.selectbox("Custom Model:", defaults.txt2vid.custom_models_list,
						                    index=defaults.txt2vid.custom_models_list.index(defaults.txt2vid.default_model),
						                    help="Select the model you want to use. This option is only available if you have custom models \
						                    on your 'models/custom' folder. The model name that will be shown here is the same as the name\
						                    the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
						                will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4") 	
						
					#st.session_state["weights_path"] = custom_model
					#else:
						#custom_model = "CompVis/stable-diffusion-v1-4"
						#st.session_state["weights_path"] = f"CompVis/{slugify(custom_model.lower())}"
						
					st.session_state.sampling_steps = st.slider("Sampling Steps", value=defaults.txt2vid.sampling_steps, min_value=10, step=10, max_value=500,
						                                    help="Number of steps between each pair of sampled points")
					st.session_state.num_inference_steps = st.slider("Inference Steps:", value=defaults.txt2vid.num_inference_steps, min_value=10,step=10, max_value=500,
						                                         help="Higher values (e.g. 100, 200 etc) can create better images.")
		
					#sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
					#sampler_name = st.selectbox("Sampling method", sampler_name_list,
								    #index=sampler_name_list.index(defaults.txt2vid.default_sampler), help="Sampling method to use. Default: k_euler")  
					scheduler_name_list = ["klms", "ddim"]
					scheduler_name = st.selectbox("Scheduler:", scheduler_name_list,
						                    index=scheduler_name_list.index(defaults.txt2vid.scheduler_name), help="Scheduler to use. Default: klms")  
					
					beta_scheduler_type_list = ["scaled_linear", "linear"]
					beta_scheduler_type = st.selectbox("Beta Schedule Type:", beta_scheduler_type_list,
						                    index=beta_scheduler_type_list.index(defaults.txt2vid.beta_scheduler_type), help="Schedule Type to use. Default: linear")  			
		
		
					#basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])
		
					#with basic_tab:
						#summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
							#help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")
		
					with st.expander("Advanced"):
						st.session_state["separate_prompts"] = st.checkbox("Create Prompt Matrix.", value=defaults.txt2vid.separate_prompts,
							                                           help="Separate multiple prompts using the `|` character, and get all combinations of them.")
						st.session_state["normalize_prompt_weights"] = st.checkbox("Normalize Prompt Weights.", 
							                                                   value=defaults.txt2vid.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
						st.session_state["save_individual_images"] = st.checkbox("Save individual images.",
							                                                 value=defaults.txt2vid.save_individual_images, help="Save each image generated before any filter or enhancement is applied.")
						st.session_state["save_video"] = st.checkbox("Save video",value=defaults.txt2vid.save_video, help="Save a video with all the images generated as frames at the end of the generation.")
						st.session_state["group_by_prompt"] = st.checkbox("Group results by prompt", value=defaults.txt2vid.group_by_prompt,
							                                          help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
						st.session_state["write_info_files"] = st.checkbox("Write Info file", value=defaults.txt2vid.write_info_files,
							                                           help="Save a file next to the image with informartion about the generation.")
						st.session_state["dynamic_preview_frequency"] = st.checkbox("Dynamic Preview Frequency", value=defaults.txt2vid.dynamic_preview_frequency,
							                                           help="This option tries to find the best value at which we can update \
						                                                   the preview image during generation while minimizing the impact it has in performance. Default: True")
						st.session_state["do_loop"] = st.checkbox("Do Loop", value=defaults.txt2vid.do_loop,
							                                  help="Do loop")
						st.session_state["save_as_jpg"] = st.checkbox("Save samples as jpg", value=defaults.txt2vid.save_as_jpg, help="Saves the images as jpg instead of png.")
		
						if GFPGAN_available:
							st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=defaults.txt2vid.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation. This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
						else:
							st.session_state["use_GFPGAN"] = False
		
						if RealESRGAN_available:
							st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=defaults.txt2vid.use_RealESRGAN,
								                                         help="Uses the RealESRGAN model to upscale the images after the generation. This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")	
							st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
						else:
							st.session_state["use_RealESRGAN"] = False
							st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"
		
						st.session_state["variant_amount"] = st.slider("Variant Amount:", value=defaults.txt2vid.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
						st.session_state["variant_seed"] = st.text_input("Variant Seed:", value=defaults.txt2vid.seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")
						st.session_state["beta_start"] = st.slider("Beta Start:", value=defaults.txt2vid.beta_start, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")
						st.session_state["beta_end"] = st.slider("Beta End:", value=defaults.txt2vid.beta_end, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")

				if generate_button:
					#print("Loading models")
					# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.		
					#load_models(False, False, False, RealESRGAN_model, CustomModel_available=CustomModel_available, custom_model=custom_model)						
					
					# run video generation
					image, seed, info, stats = txt2vid(prompts=prompt, gpu=defaults.general.gpu,
						                   num_steps=st.session_state.sampling_steps, max_frames=int(st.session_state.max_frames),
						                   num_inference_steps=st.session_state.num_inference_steps,
						                   cfg_scale=cfg_scale,do_loop=st.session_state["do_loop"],
						                   seeds=seed, quality=100, eta=0.0, width=width,
						                   height=height, weights_path=custom_model, scheduler=scheduler_name,
						                   disable_tqdm=False, beta_start=st.session_state["beta_start"], beta_end=st.session_state["beta_end"],
						                   beta_schedule=beta_scheduler_type)
					    
					#message.success('Done!', icon="✅")
					message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")
		
					#except (StopException, KeyError):
						#print(f"Received Streamlit StopException")
		
					# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
					# use the current col2 first tab to show the preview_img and update it as its generated.
					#preview_image.image(output_images)		

	#
	elif tabs == 'Model Manager':
		#search = st.text_input(label="Search", placeholder="Type the name of the model you want to search for.", help="")

		csvString = f"""
		 ,Stable Diffusion v1.4       , ./models/ldm/stable-diffusion-v1               , https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media                  
		 ,GFPGAN v1.3                 , ./src/gfpgan/experiments/pretrained_models     , https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth                     
		 ,RealESRGAN_x4plus           , ./src/realesrgan/experiments/pretrained_models , https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth            
		 ,RealESRGAN_x4plus_anime_6B  , ./src/realesrgan/experiments/pretrained_models , https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth 
		 ,Waifu Diffusion v1.2        , ./models/custom                                , http://wd.links.sd:8880/wd-v1-2-full-ema.ckpt
		 ,TrinArt Stable Diffusion v2 , ./models/custom                                , https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step115000.ckpt
		"""
		colms = st.columns((1, 3, 5, 5))
		columns = ["№",'Model Name','Save Location','Download Link']
		
		# Convert String into StringIO
		csvStringIO = StringIO(csvString)
		df = pd.read_csv(csvStringIO, sep=",", header=None, names=columns)		
		
		for col, field_name in zip(colms, columns):
			# table header
			col.write(field_name)
		
		for x, model_name in enumerate(df["Model Name"]):
			col1, col2, col3, col4 = st.columns((1, 3, 4, 6))
			col1.write(x)  # index
			col2.write(df['Model Name'][x])
			col3.write(df['Save Location'][x])
			col4.write(df['Download Link'][x])
				

	elif tabs == 'Settings':	
		import Settings
			
		st.write("Settings")

if __name__ == '__main__':
	layout()     

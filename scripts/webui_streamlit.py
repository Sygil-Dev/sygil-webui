import warnings
import importlib
import streamlit as st
from streamlit import StopException, StreamlitAPIException
import sd_utils as SDutils
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


# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(defaults.general.gpu)






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
		model_wrap_cfg = SDutils.CFGDenoiser(self.model_wrap)
		samples_ddim = None
		samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas,
                                                                              extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,
                                                                                          'cond_scale': unconditional_guidance_scale}, disable=False, callback=SDutils.generation_callback)
		#
		return samples_ddim, None












#@retry(RuntimeError, tries=3)




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


	#txt2img_tab, img2img_tab, txt2video, postprocessing_tab = st.tabs(["Text-to-Image Unified", "Image-to-Image Unified", "Text-to-Video","Post-Processing"])
	# scan plugins folder for plugins and add them to the st.tabs
	plugins = {}
	for plugin in os.listdir("scripts/plugins"):
		if plugin.endswith(".py"):
			# return the description of the plugin
			pluginModule = importlib.import_module(f"scripts.plugins.{plugin[:-3]}")
			importlib.reload(pluginModule)
			pluginDescription = pluginModule.PluginInfo.description
			pluginPriority = pluginModule.PluginInfo.displayPriority
			pluginIsTab = pluginModule.PluginInfo.isTab
			# if the plugin is a tab, add it to the tabs
			if pluginIsTab:
				plugins[pluginDescription] = [pluginModule, pluginPriority]
	
	#print(plugins)
	#print(pluginTabs)
	#print(plugins)
	# sort the plugins by priority
	plugins = {k: v for k, v in sorted(plugins.items(), key=lambda x: x[1][1])}
	pluginTabs = st.tabs(plugins)
	increment = 0
	for k in plugins.keys():
		with pluginTabs[increment]:
				plugins[k][0].layoutFunc()
				increment += 1

			#print(plugin)
			# print(plugin.description)
			#plugin.layout
	
if __name__ == '__main__':
	layout()     
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
if (os.path.exists("userconfig_streamlit.yaml")):
	user_defaults = OmegaConf.load("userconfig_streamlit.yaml");
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

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(defaults.general.gpu)

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
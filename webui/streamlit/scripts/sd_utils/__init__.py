# This file is part of sygil-webui (https://github.com/Sygil-Dev/sygil-webui/).

# Copyright 2022 Sygil-Dev team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# base webui import and utils.
#from webui_streamlit import st
import hydralit as st

# streamlit imports
from streamlit.runtime.scriptrunner import StopException
#from streamlit.runtime.scriptrunner import script_run_context

#streamlit components section
from streamlit_server_state import server_state, server_state_lock, no_rerun
import hydralit_components as hc
from hydralit import HydraHeadApp
import streamlit_nested_layout
#from streamlitextras.threader import lock, trigger_rerun, \
                                     #streamlit_thread, get_thread, \
                                     #last_trigger_time

#other imports

import warnings
import json

import base64, cv2
import os, sys, re, random, datetime, time, math, toml
import gc
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo
from scipy import integrate
import torch
from torchdiffeq import odeint
import k_diffusion as K
import math, requests
import mimetypes
import numpy as np
import pynvml
import threading
import torch, torchvision
from torch import autocast
from torchvision import transforms
import torch.nn as nn
from omegaconf import OmegaConf
import yaml
from pathlib import Path
from contextlib import nullcontext
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from retry import retry
from slugify import slugify
import skimage
import piexif
import piexif.helper
from tqdm import trange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import ismap
#from abc import ABC, abstractmethod
from io import BytesIO
from packaging import version
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutup

#import librosa
from nataili.util.logger import logger, set_logger_verbosity, quiesce_logger
from nataili.esrgan import esrgan


#try:
    #from realesrgan import RealESRGANer
    #from basicsr.archs.rrdbnet_arch import RRDBNet
#except ImportError as e:
    #logger.error("You tried to import realesrgan without having it installed properly. To install Real-ESRGAN, run:\n\n"
        #"pip install realesrgan")

# Temp imports
#from basicsr.utils.registry import ARCH_REGISTRY


# end of imports
#---------------------------------------------------------------------------------------------------------------

# remove all the annoying python warnings.
shutup.please()

# the following lines should help fixing an issue with nvidia 16xx cards.
if "defaults" in st.session_state:
    if st.session_state["defaults"].general.use_cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True 

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

# disable diffusers telemetry
os.environ["DISABLE_TELEMETRY"] = "YES"

# remove some annoying deprecation warnings that show every now and then.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

# The model manager loads and unloads the SD models and has features to download them or find their location
#model_manager = ModelManager()

def load_configs():
    if not "defaults" in st.session_state:
        st.session_state["defaults"] = {}

    st.session_state["defaults"] = OmegaConf.load("configs/webui/webui_streamlit.yaml")

    if (os.path.exists("configs/webui/userconfig_streamlit.yaml")):
        user_defaults = OmegaConf.load("configs/webui/userconfig_streamlit.yaml")

        if "version" in user_defaults.general:
            if version.parse(user_defaults.general.version) < version.parse(st.session_state["defaults"].general.version):
                logger.error("The version of the user config file is older than the version on the defaults config file. "
                             "This means there were big changes we made on the config."
                         "We are removing this file and recreating it from the defaults in order to make sure things work properly.")
                os.remove("configs/webui/userconfig_streamlit.yaml")
                st.experimental_rerun()
        else:
            logger.error("The version of the user config file is older than the version on the defaults config file. "
                         "This means there were big changes we made on the config."
                         "We are removing this file and recreating it from the defaults in order to make sure things work properly.")
            os.remove("configs/webui/userconfig_streamlit.yaml")
            st.experimental_rerun()

        try:
            st.session_state["defaults"] = OmegaConf.merge(st.session_state["defaults"], user_defaults)
        except KeyError:
            st.experimental_rerun()
    else:
        OmegaConf.save(config=st.session_state.defaults, f="configs/webui/userconfig_streamlit.yaml")
        loaded = OmegaConf.load("configs/webui/userconfig_streamlit.yaml")
        assert st.session_state.defaults == loaded

    if (os.path.exists(".streamlit/config.toml")):
        st.session_state["streamlit_config"] = toml.load(".streamlit/config.toml")

    #if st.session_state["defaults"].daisi_app.running_on_daisi_io:
        #if os.path.exists("scripts/modeldownload.py"):
            #import modeldownload
            #modeldownload.updateModels()

    if "keep_all_models_loaded" in st.session_state.defaults.general:
        with server_state_lock["keep_all_models_loaded"]:
            server_state["keep_all_models_loaded"] = st.session_state["defaults"].general.keep_all_models_loaded
    else:
        st.session_state["defaults"].general.keep_all_models_loaded = False
        with server_state_lock["keep_all_models_loaded"]:
            server_state["keep_all_models_loaded"] = st.session_state["defaults"].general.keep_all_models_loaded

load_configs()

#
#if st.session_state["defaults"].debug.enable_hydralit:
    #navbar_theme = {'txc_inactive': '#FFFFFF','menu_background':'#0e1117','txc_active':'black','option_active':'red'}
    #app = st.HydraApp(title='Stable Diffusion WebUI', favicon="", use_cookie_cache=False, sidebar_state="expanded", layout="wide", navbar_theme=navbar_theme,
                      #hide_streamlit_markers=False, allow_url_nav=True , clear_cross_app_sessions=False, use_loader=False)
#else:
    #app = None

#
grid_format = st.session_state["defaults"].general.save_format
grid_lossless = False
grid_quality = st.session_state["defaults"].general.grid_quality
if grid_format == 'png':
    grid_ext = 'png'
    grid_format = 'png'
elif grid_format in ['jpg', 'jpeg']:
    grid_quality = int(grid_format) if len(grid_format) > 1 else 100
    grid_ext = 'jpg'
    grid_format = 'jpeg'
elif grid_format[0] == 'webp':
    grid_quality = int(grid_format) if len(grid_format) > 1 else 100
    grid_ext = 'webp'
    grid_format = 'webp'
    if grid_quality < 0: # e.g. webp:-100 for lossless mode
        grid_lossless = True
        grid_quality = abs(grid_quality)

#
save_format = st.session_state["defaults"].general.save_format
save_lossless = False
save_quality = 100
if save_format == 'png':
    save_ext = 'png'
    save_format = 'png'
elif save_format in ['jpg', 'jpeg']:
    save_quality = int(save_format) if len(save_format) > 1 else 100
    save_ext = 'jpg'
    save_format = 'jpeg'
elif save_format == 'webp':
    save_quality = int(save_format) if len(save_format) > 1 else 100
    save_ext = 'webp'
    save_format = 'webp'
    if save_quality < 0: # e.g. webp:-100 for lossless mode
        save_lossless = True
        save_quality = abs(save_quality)

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(st.session_state["defaults"].general.gpu)


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

def set_page_title(title):
    """
    Simple function to allows us to change the title dynamically.
    Normally you can use `st.set_page_config` to change the title but it can only be used once per app.
    """

    st.sidebar.markdown(unsafe_allow_html=True, body=f"""
                            <iframe height=0 srcdoc="<script>
                            const title = window.parent.document.querySelector('title') \

                            const oldObserver = window.parent.titleObserver
                            if (oldObserver) {{
                            oldObserver.disconnect()
                            }} \

                            const newObserver = new MutationObserver(function(mutations) {{
                            const target = mutations[0].target
                            if (target.text !== '{title}') {{
                            target.text = '{title}'
                            }}
                            }}) \

                            newObserver.observe(title, {{ childList: true }})
                            window.parent.titleObserver = newObserver \

                            title.text = '{title}'
                            </script>" />
                            """)

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
            logger.debug(f"[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n")
            return
        logger.info(f"[{self.name}] Recording memory usage...\n")
        # Missing context
        #handle = pynvml.nvmlDeviceGetHandleByIndex(st.session_state['defaults'].general.gpu)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # logger.info(self.max_usage)
            time.sleep(0.1)
        logger.info(f"[{self.name}] Stopped recording.\n")
        pynvml.nvmlShutdown()

    def read(self):
        return self.max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

#
def custom_models_available():
    with server_state_lock["custom_models"]:
        #
        # Allow for custom models to be used instead of the default one,
        # an example would be Waifu-Diffusion or any other fine tune of stable diffusion
        server_state["custom_models"]:sorted = []

        for root, dirs, files in os.walk(os.path.join("models", "custom")):
            for file in files:
                if os.path.splitext(file)[1] == '.ckpt':
                    server_state["custom_models"].append(os.path.splitext(file)[0])

        with server_state_lock["CustomModel_available"]:
            if len(server_state["custom_models"]) > 0:
                server_state["CustomModel_available"] = True
                server_state["custom_models"].append("Stable Diffusion v1.5")
            else:
                server_state["CustomModel_available"] = False

#
def GFPGAN_available():
    #with server_state_lock["GFPGAN_models"]:
    #

    st.session_state["GFPGAN_models"]:sorted = []
    model = st.session_state["defaults"].model_manager.models.gfpgan

    files_available = 0

    for file in model['files']:
        if "save_location" in model['files'][file]:
            if os.path.exists(os.path.join(model['files'][file]['save_location'], model['files'][file]['file_name'] )):
                files_available += 1

        elif os.path.exists(os.path.join(model['save_location'], model['files'][file]['file_name'] )):
            base_name = os.path.splitext(model['files'][file]['file_name'])[0]
            if "GFPGANv" in base_name:
                st.session_state["GFPGAN_models"].append(base_name)
            files_available += 1

    # we need to show the other models from previous verions that we have on the
    # same directory in case we want to see how they perform vs each other.
    for root, dirs, files in os.walk(st.session_state['defaults'].general.GFPGAN_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pth':
                if os.path.splitext(file)[0] not in st.session_state["GFPGAN_models"]:
                    st.session_state["GFPGAN_models"].append(os.path.splitext(file)[0])


    if len(st.session_state["GFPGAN_models"]) > 0 and files_available == len(model['files']):
        st.session_state["GFPGAN_available"] = True
    else:
        st.session_state["GFPGAN_available"] = False
        st.session_state["use_GFPGAN"] = False
        st.session_state["GFPGAN_model"] = "GFPGANv1.4"

#
def RealESRGAN_available():
    #with server_state_lock["RealESRGAN_models"]:
    
    st.session_state["RealESRGAN_models"]:sorted = []
    model = st.session_state["defaults"].model_manager.models.realesrgan
    for file in model['files']:
        if os.path.exists(os.path.join(model['save_location'], model['files'][file]['file_name'] )):
            base_name = os.path.splitext(model['files'][file]['file_name'])[0]
            st.session_state["RealESRGAN_models"].append(base_name)

    if len(st.session_state["RealESRGAN_models"]) > 0:
        st.session_state["RealESRGAN_available"] = True
    else:
        st.session_state["RealESRGAN_available"] = False
        st.session_state["use_RealESRGAN"] = False
        st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"
#
def LDSR_available():
    st.session_state["LDSR_models"]:sorted = []
    files_available = 0
    model = st.session_state["defaults"].model_manager.models.ldsr
    for file in model['files']:
        if os.path.exists(os.path.join(model['save_location'], model['files'][file]['file_name'] )):
            base_name = os.path.splitext(model['files'][file]['file_name'])[0]
            extension = os.path.splitext(model['files'][file]['file_name'])[1]
            if extension == ".ckpt":
                st.session_state["LDSR_models"].append(base_name)
            files_available += 1
    if files_available == len(model['files']):
        st.session_state["LDSR_available"] = True
    else:
        st.session_state["LDSR_available"] = False
        st.session_state["use_LDSR"] = False
        st.session_state["LDSR_model"] = "model"

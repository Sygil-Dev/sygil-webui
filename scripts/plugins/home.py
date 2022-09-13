#home plugin
from multiprocessing import AuthenticationError
from unicodedata import name
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
from .. import sd_utils as SDutils
# we use python-slugify to make the filenames safe for windows and linux, its better than doing it manually
# install it with 'pip install python-slugify'
from slugify import slugify
try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass
defaults = OmegaConf.load("configs/webui/webui_streamlit.yaml")

class PluginInfo():
        plugname = "home"
        description = "Home"
        isTab = True
        displayPriority = 0
        
def getLatestGeneratedImagesFromPath():
    #get the latest images from the generated images folder
    #get the path to the generated images folder
    generatedImagesPath = os.path.join(os.getcwd(),'outputs')
    #get all the files from the folders and subfolders
    files = []
    #get the laest 10 images from the output folder without walking the subfolders
    for r, d, f in os.walk(generatedImagesPath):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    #sort the files by date
    files.sort(key=os.path.getmtime)
    #reverse the list so the latest images are first
    for f in files:
        img = Image.open(f)
        files[files.index(f)] = img
    #get the latest 10 files
    #get all the files with the .png or .jpg extension
    #sort files by date
    #get the latest 10 files
    latestFiles = files[-10:]
    #reverse the list
    latestFiles.reverse()
    return latestFiles
def getImagesFromLexica():
    #scrape images from lexica.art
    #get the html from the page
    #get the html with cookies and javascript
    apiEndpoint = r'https://lexica.art/api/trpc/prompts.infinitePrompts?batch=1&input=%7B%220%22%3A%7B%22json%22%3A%7B%22limit%22%3A10%2C%22text%22%3A%22%22%2C%22cursor%22%3A10%7D%7D%7D'
    #REST API call
    # 
    from requests_html import HTMLSession
    session = HTMLSession()

    response = session.get(apiEndpoint)
    #req = requests.Session()
    #req.headers['user-agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
    #response = req.get(apiEndpoint)
    print(response.status_code)
    print(response.text)
    #get the json from the response
    #json = response.json()
    #get the prompts from the json
    print(response)
    #session = requests.Session()
    #parseEndpointJson = session.get(apiEndpoint,headers=headers,verify=False)
    #print(parseEndpointJson)
    #print('test2')
    #page = requests.get("https://lexica.art/", headers={'User-Agent': 'Mozilla/5.0'})
    #parse the html
    #soup = BeautifulSoup(page.content, 'html.parser')
    #find all the images
    #print(soup)
    #images = soup.find_all('alt-image')
    #create a list to store the image urls
    image_urls = []
    #loop through the images
    for image in images:
        #get the url
        image_url = image['src']
        #add it to the list
        image_urls.append('http://www.lexica.art/'+image_url)
    #return the list
    print(image_urls)
    return image_urls
def layoutFunc():
    #streamlit home page layout
    #center the title
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome, let's make some 🎨</h1>", unsafe_allow_html=True)
    #make a gallery of images
    #st.markdown("<h2 style='text-align: center; color: white;'>Gallery</h2>", unsafe_allow_html=True)
    #create a gallery of images using columns
    #col1, col2, col3 = st.columns(3)
    #load the images
    #create 3 columns
    # create a tab for the gallery
    #st.markdown("<h2 style='text-align: center; color: white;'>Gallery</h2>", unsafe_allow_html=True)
    #st.markdown("<h2 style='text-align: center; color: white;'>Gallery</h2>", unsafe_allow_html=True)
    history_tab, discover_tabs, settings_tab = st.tabs(["History","Discover","Settings"])
    with discover_tabs:
        st.markdown("<h1 style='text-align: center; color: white;'>Soon :)</h1>", unsafe_allow_html=True)
    with settings_tab:
        st.markdown("<h1 style='text-align: center; color: white;'>Soon :)</h1>", unsafe_allow_html=True)
    with history_tab:
        placeholder = st.empty()
        

        
        latestImages = getLatestGeneratedImagesFromPath()
        st.session_state['latestImages'] = latestImages
        
        #populate the 3 images per column
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1_cont = st.container()
            col2_cont = st.container()
            col3_cont = st.container()
            with col1_cont:
                with col1:
                    st.image(st.session_state['latestImages'][0])
                    st.image(st.session_state['latestImages'][3])
                    st.image(st.session_state['latestImages'][6])
            with col2_cont:
                with col2:
                    st.image(st.session_state['latestImages'][1])
                    st.image(st.session_state['latestImages'][4])
                    st.image(st.session_state['latestImages'][7])
            with col3_cont:
                with col3:
                    st.image(st.session_state['latestImages'][2])
                    st.image(st.session_state['latestImages'][5])
                    st.image(st.session_state['latestImages'][8])
        st.session_state['historyTab'] = [history_tab,col1,col2,col3,placeholder,col1_cont,col2_cont,col3_cont]
    #display the images
    #add a button to the gallery
    #st.markdown("<h2 style='text-align: center; color: white;'>Try it out</h2>", unsafe_allow_html=True)
    #create a button to the gallery
    #if st.button("Try it out"):
        #if the button is clicked, go to the gallery
        #st.experimental_rerun()

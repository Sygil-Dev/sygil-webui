#home plugin
from distutils.command.upload import upload
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
from bs4 import BeautifulSoup
from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
from streamlit.elements import image as STImage
try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass
defaults = OmegaConf.load("configs/webui/webui_streamlit.yaml")

class PluginInfo():
        plugname = "imglab"
        description = "Image Lab"
        isTab = True
        displayPriority = 3
        
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
def changeImage():
        #change the image in the image holder
        #check if the file is not empty
        if len(st.session_state['uploaded_file']) > 0:
            #read the file
            print('test2')
            uploaded = st.session_state['uploaded_file'][0].read()
            #show the image in the image holder
            st.session_state['previewImg'].empty()
            st.session_state['previewImg'].image(uploaded,use_column_width=True)
def createHTMLGallery(images):
    html3 = """
        <div class="gallery-history" style="
    display: flex;
    flex-wrap: wrap;
	align-items: flex-start;">
        """
    mkdwn_array = []
    for i in images:
        bImg = i.read()
        i = Image.save(bImg, 'PNG')
        width, height = i.size
        #get random number for the id
        image_id = "%s" % (str(images.index(i)))
        (data, mimetype) = STImage._normalize_to_bytes(bImg.getvalue(), width, 'auto')
        this_file = in_memory_file_manager.add(data, mimetype, image_id)
        img_str = this_file.url
        #img_str = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
        #get image size
        
        #make sure the image is not bigger then 150px but keep the aspect ratio
        if width > 150:
            height = int(height * (150/width))
            width = 150
        if height > 150:
            width = int(width * (150/height))
            height = 150

        #mkdwn = f"""<img src="{img_str}" alt="Image" with="200" height="200" />"""
        mkdwn = f'''<div class="gallery" style="margin: 3px;" >
<a href="{img_str}">
<img src="{img_str}" alt="Image" width="{width}" height="{height}">
</a>
</div>
'''
        mkdwn_array.append(mkdwn)
    html3 += "".join(mkdwn_array)
    html3 += '</div>'
    return html3
def layoutFunc():
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state['uploaded_file'] = st.file_uploader("Choose an image or images", type=["png", "jpg", "jpeg"],accept_multiple_files=True,on_change=changeImage)
            if 'previewImg' not in st.session_state:
                st.session_state['previewImg'] = st.empty()
            else:
                if len(st.session_state['uploaded_file']) > 0:
                    st.session_state['previewImg'].empty()
                    st.session_state['previewImg'].image(st.session_state['uploaded_file'][0],use_column_width=True)
                else:
                    st.session_state['previewImg'] = st.empty()
            

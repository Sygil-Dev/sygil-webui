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

#define a class to get the plugin info
class PluginInfo():
        plugname = "img2img"
        description = "Image to Image"
        isTab = True
        displayPriority = 2
        


if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
	GFPGAN_available = True
else:
	GFPGAN_available = False

if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{defaults.general.RealESRGAN_model}.pth")):
	RealESRGAN_available = True
else:
	RealESRGAN_available = False	
	
	

def func(prompt: str = '', init_info: any = None, init_info_mask: any = None, mask_mode: int = 0, mask_blur_strength: int = 3, 
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
	seed = SDutils.seed_to_int(seed)

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
		sampler = SDutils.KDiffusionSampler(st.session_state["model"],'dpm_2_ancestral')
	elif sampler_name == 'k_dpm_2':
		sampler = SDutils.KDiffusionSampler(st.session_state["model"],'dpm_2')
	elif sampler_name == 'k_euler_a':
		sampler = SDutils.KDiffusionSampler(st.session_state["model"],'euler_ancestral')
	elif sampler_name == 'k_euler':
		sampler = SDutils.KDiffusionSampler(st.session_state["model"],'euler')
	elif sampler_name == 'k_heun':
		sampler = SDutils.KDiffusionSampler(st.session_state["model"],'heun')
	elif sampler_name == 'k_lms':
		sampler = SDutils.KDiffusionSampler(st.session_state["model"],'lms')
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
			model_wrap_cfg = SDutils.CFGMaskedDenoiser(sampler.model_wrap)
			samples_ddim = K.sampling.__dict__[f'sample_{sampler.get_sampler_name()}'](model_wrap_cfg, xi, sigma_sched,
                                                                                                   extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,
                                                                                                               'cond_scale': cfg_scale, 'mask': z_mask, 'x0': x0, 'xi': xi}, disable=False,
                                                                                                   callback=SDutils.generation_callback)
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

			output_images, seed, info, stats = SDutils.process_images(
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
				seed = SDutils.seed_to_int(None)

			denoising_strength = max(denoising_strength * 0.95, 0.1)
			history.append(init_img)

		output_images = history
		seed = initial_seed

	else:
		output_images, seed, info, stats = SDutils.process_images(
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
def layoutFunc():
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
				SDutils.load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model)                
				if uploaded_images:
					image = Image.open(uploaded_images).convert('RGB')
					new_img = image.resize((width, height))
					#img_array = np.array(image) # if you want to pass it to OpenCV

					try:
						output_images, seed, info, stats = func(prompt=prompt, init_info=new_img, ddim_steps=st.session_state["sampling_steps"],
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

#on import run init

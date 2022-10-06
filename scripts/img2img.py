# This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

# Copyright 2022 sd-webui team.
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
from sd_utils import *

# streamlit imports
from streamlit import StopException

#other imports
import cv2
from PIL import Image, ImageOps
import torch
import k_diffusion as K
import numpy as np
import time
import torch
import skimage
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
# Temp imports


# end of imports
#---------------------------------------------------------------------------------------------------------------


try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass

def img2img(prompt: str = '', init_info: any = None, init_info_mask: any = None, mask_mode: int = 0, mask_blur_strength: int = 3,
	    mask_restore: bool = False, ddim_steps: int = 50, sampler_name: str = 'DDIM',
	    n_iter: int = 1,  cfg_scale: float = 7.5, denoising_strength: float = 0.8,
	    seed: int = -1, noise_mode: int = 0, find_noise_steps: str = "", height: int = 512, width: int = 512, resize_mode: int = 0, fp = None,
	    variant_amount: float = None, variant_seed: int = None, ddim_eta:float = 0.0,
	    write_info_files:bool = True, separate_prompts:bool = False, normalize_prompt_weights:bool = True,
	    save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
	    save_as_jpg: bool = True, use_GFPGAN: bool = True, GFPGAN_model: str = 'GFPGANv1.4',
		use_RealESRGAN: bool = True, RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B",
		use_LDSR: bool = True, LDSR_model: str = "model",
		loopback: bool = False,
	    random_seed_loopback: bool = False
	    ):

	outpath = st.session_state['defaults'].general.outdir_img2img
	seed = seed_to_int(seed)

	batch_size = 1

	if sampler_name == 'PLMS':
		sampler = PLMSSampler(server_state["model"])
	elif sampler_name == 'DDIM':
		sampler = DDIMSampler(server_state["model"])
	elif sampler_name == 'k_dpm_2_a':
		sampler = KDiffusionSampler(server_state["model"],'dpm_2_ancestral')
	elif sampler_name == 'k_dpm_2':
		sampler = KDiffusionSampler(server_state["model"],'dpm_2')
	elif sampler_name == 'k_euler_a':
		sampler = KDiffusionSampler(server_state["model"],'euler_ancestral')
	elif sampler_name == 'k_euler':
		sampler = KDiffusionSampler(server_state["model"],'euler')
	elif sampler_name == 'k_heun':
		sampler = KDiffusionSampler(server_state["model"],'heun')
	elif sampler_name == 'k_lms':
		sampler = KDiffusionSampler(server_state["model"],'lms')
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

		noise_rgb = get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
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
			mask = torch.from_numpy(mask).to(server_state["device"])

		if st.session_state['defaults'].general.optimized:
			server_state["modelFS"].to(server_state["device"] )

		init_image = 2. * image - 1.
		init_image = init_image.to(server_state["device"])
		init_latent = (server_state["model"] if not st.session_state['defaults'].general.optimized else server_state["modelFS"]).get_first_stage_encoding((server_state["model"]  if not st.session_state['defaults'].general.optimized else server_state["modelFS"]).encode_first_stage(init_image))  # move to latent space

		if st.session_state['defaults'].general.optimized:
			mem = torch.cuda.memory_allocated()/1e6
			server_state["modelFS"].to("cpu")
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
			z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc_steps]*batch_size).to(server_state["device"] ))

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

			# RealESRGAN can only run on the final iteration
			is_final_iteration = i == n_iter - 1

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
				GFPGAN_model=GFPGAN_model,
				use_RealESRGAN=use_RealESRGAN and is_final_iteration, # Forcefully disable upscaling when using loopback
				realesrgan_model_name=RealESRGAN_model,
				use_LDSR=use_LDSR,
				LDSR_model_name=LDSR_model,
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

			input_image = init_img
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
				if mask_restore is True and init_mask is not None:
					color_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
					color_mask = color_mask.convert('L')
					source_image = input_image.convert('RGB')
					target_image = init_img.convert('RGB')

					init_img = Image.composite(source_image, target_image, color_mask)

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
			GFPGAN_model=GFPGAN_model,
			use_RealESRGAN=use_RealESRGAN,
			realesrgan_model_name=RealESRGAN_model,
			use_LDSR=use_LDSR,
			LDSR_model_name=LDSR_model,
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

#
def layout():
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
			custom_models_available()
			if server_state["CustomModel_available"]:
				st.session_state["custom_model"] = st.selectbox("Custom Model:", server_state["custom_models"],
									    index=server_state["custom_models"].index(st.session_state['defaults'].general.default_model),
							    help="Select the model you want to use. This option is only available if you have custom models \
							    on your 'models/custom' folder. The model name that will be shown here is the same as the name\
							    the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
							    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
			else:
				st.session_state["custom_model"] = "Stable Diffusion v1.4"


			st.session_state["sampling_steps"] = st.number_input("Sampling Steps", value=st.session_state['defaults'].img2img.sampling_steps.value,
																 min_value=st.session_state['defaults'].img2img.sampling_steps.min_value,
																 step=st.session_state['defaults'].img2img.sampling_steps.step)

			sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
			st.session_state["sampler_name"] = st.selectbox("Sampling method",sampler_name_list,
									index=sampler_name_list.index(st.session_state['defaults'].img2img.sampler_name), help="Sampling method to use.")

			width = st.slider("Width:", min_value=st.session_state['defaults'].img2img.width.min_value, max_value=st.session_state['defaults'].img2img.width.max_value,
									  value=st.session_state['defaults'].img2img.width.value, step=st.session_state['defaults'].img2img.width.step)
			height = st.slider("Height:", min_value=st.session_state['defaults'].img2img.height.min_value, max_value=st.session_state['defaults'].img2img.height.max_value,
									   value=st.session_state['defaults'].img2img.height.value, step=st.session_state['defaults'].img2img.height.step)
			seed = st.text_input("Seed:", value=st.session_state['defaults'].img2img.seed, help=" The seed to use, if left blank a random seed will be generated.")

			cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=st.session_state['defaults'].img2img.cfg_scale.min_value,
											  max_value=st.session_state['defaults'].img2img.cfg_scale.max_value, value=st.session_state['defaults'].img2img.cfg_scale.value,
											  step=st.session_state['defaults'].img2img.cfg_scale.step, help="How strongly the image should follow the prompt.")

			st.session_state["denoising_strength"] = st.slider("Denoising Strength:", value=st.session_state['defaults'].img2img.denoising_strength.value,
																		   min_value=st.session_state['defaults'].img2img.denoising_strength.min_value,
														   max_value=st.session_state['defaults'].img2img.denoising_strength.max_value,
														   step=st.session_state['defaults'].img2img.denoising_strength.step)


			mask_expander = st.empty()
			with mask_expander.expander("Mask"):
				mask_mode_list = ["Mask", "Inverted mask", "Image alpha"]
				mask_mode = st.selectbox("Mask Mode", mask_mode_list,
									 help="Select how you want your image to be masked.\"Mask\" modifies the image where the mask is white.\n\
									 \"Inverted mask\" modifies the image where the mask is black. \"Image alpha\" modifies the image where the image is transparent."
									 )
				mask_mode = mask_mode_list.index(mask_mode)


				noise_mode_list = ["Seed", "Find Noise", "Matched Noise", "Find+Matched Noise"]
				noise_mode = st.selectbox(
							"Noise Mode", noise_mode_list,
							help=""
						)
				noise_mode = noise_mode_list.index(noise_mode)
				find_noise_steps = st.slider("Find Noise Steps", value=st.session_state['defaults'].img2img.find_noise_steps.value,
											 min_value=st.session_state['defaults'].img2img.find_noise_steps.min_value, max_value=st.session_state['defaults'].img2img.find_noise_steps.max_value,
											 step=st.session_state['defaults'].img2img.find_noise_steps.step)

			with st.expander("Batch Options"):
				st.session_state["batch_count"] = int(st.text_input("Batch count.", value=st.session_state['defaults'].img2img.batch_count.value,
																help="How many iterations or batches of images to generate in total."))

				st.session_state["batch_size"] = int(st.text_input("Batch size", value=st.session_state.defaults.img2img.batch_size.value,
				                            help="How many images are at once in a batch.\
				                            It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
                                            Default: 1"))

			with st.expander("Preview Settings"):
				st.session_state["update_preview"] = st.session_state["defaults"].general.update_preview
				st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=st.session_state['defaults'].img2img.update_preview_frequency,
																					 help="Frequency in steps at which the the preview image is updated. By default the frequency \
															  is set to 1 step.")
			#
			with st.expander("Advanced"):
				with st.expander("Output Settings"):
					separate_prompts = st.checkbox("Create Prompt Matrix.", value=st.session_state['defaults'].img2img.separate_prompts,
						                       help="Separate multiple prompts using the `|` character, and get all combinations of them.")
					normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=st.session_state['defaults'].img2img.normalize_prompt_weights,
						                           help="Ensure the sum of all weights add up to 1.0")
					loopback = st.checkbox("Loopback.", value=st.session_state['defaults'].img2img.loopback, help="Use images from previous batch when creating next batch.")
					random_seed_loopback = st.checkbox("Random loopback seed.", value=st.session_state['defaults'].img2img.random_seed_loopback, help="Random loopback seed")
					img2img_mask_restore = st.checkbox("Only modify regenerated parts of image",
						                               value=st.session_state['defaults'].img2img.mask_restore,
						                               help="Enable to restore the unmasked parts of the image with the input, may not blend as well but preserves detail")
					save_individual_images = st.checkbox("Save individual images.", value=st.session_state['defaults'].img2img.save_individual_images,
						                         help="Save each image generated before any filter or enhancement is applied.")
					save_grid = st.checkbox("Save grid",value=st.session_state['defaults'].img2img.save_grid, help="Save a grid with all the images generated into a single image.")
					group_by_prompt = st.checkbox("Group results by prompt", value=st.session_state['defaults'].img2img.group_by_prompt,
						                      help="Saves all the images with the same prompt into the same folder. \
						                      When using a prompt matrix each prompt combination will have its own folder.")
					write_info_files = st.checkbox("Write Info file", value=st.session_state['defaults'].img2img.write_info_files,
						                       help="Save a file next to the image with informartion about the generation.")
					save_as_jpg = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].img2img.save_as_jpg, help="Saves the images as jpg instead of png.")

				#
				# check if GFPGAN, RealESRGAN and LDSR are available.
				if "GFPGAN_available" not in st.session_state:
					GFPGAN_available()

				if "RealESRGAN_available" not in st.session_state:
					RealESRGAN_available()

				if "LDSR_available" not in st.session_state:
					LDSR_available()

				if st.session_state["GFPGAN_available"] or st.session_state["RealESRGAN_available"] or st.session_state["LDSR_available"]:
					with st.expander("Post-Processing"):
						face_restoration_tab, upscaling_tab = st.tabs(["Face Restoration", "Upscaling"])
						with face_restoration_tab:
							# GFPGAN used for face restoration
							if st.session_state["GFPGAN_available"]:
								#with st.expander("Face Restoration"):
								#if st.session_state["GFPGAN_available"]:
								#with st.expander("GFPGAN"):
								st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].img2img.use_GFPGAN,
																							 help="Uses the GFPGAN model to improve faces after the generation.\
																							 This greatly improve the quality and consistency of faces but uses\
																							 extra VRAM. Disable if you need the extra VRAM.")

								st.session_state["GFPGAN_model"] = st.selectbox("GFPGAN model", st.session_state["GFPGAN_models"],
																								index=st.session_state["GFPGAN_models"].index(st.session_state['defaults'].general.GFPGAN_model))

								#st.session_state["GFPGAN_strenght"] = st.slider("Effect Strenght", min_value=1, max_value=100, value=1, step=1, help='')

							else:
								st.session_state["use_GFPGAN"] = False

						with upscaling_tab:
							st.session_state['us_upscaling'] = st.checkbox("Use Upscaling", value=st.session_state['defaults'].img2img.use_upscaling)

							# RealESRGAN and LDSR used for upscaling.
							if st.session_state["RealESRGAN_available"] or st.session_state["LDSR_available"]:

								upscaling_method_list = []
								if st.session_state["RealESRGAN_available"]:
									upscaling_method_list.append("RealESRGAN")
								if st.session_state["LDSR_available"]:
									upscaling_method_list.append("LDSR")

								st.session_state["upscaling_method"] = st.selectbox("Upscaling Method", upscaling_method_list,
																								index=upscaling_method_list.index(st.session_state['defaults'].general.upscaling_method))

								if st.session_state["RealESRGAN_available"]:
									with st.expander("RealESRGAN"):
										if st.session_state["upscaling_method"] == "RealESRGAN" and st.session_state['us_upscaling']:
											st.session_state["use_RealESRGAN"] = True
										else:
											st.session_state["use_RealESRGAN"] = False

										st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", st.session_state["RealESRGAN_models"],
																							index=st.session_state["RealESRGAN_models"].index(st.session_state['defaults'].general.RealESRGAN_model))
								else:
									st.session_state["use_RealESRGAN"] = False
									st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"


								#
								if st.session_state["LDSR_available"]:
									with st.expander("LDSR"):
										if st.session_state["upscaling_method"] == "LDSR" and st.session_state['us_upscaling']:
											st.session_state["use_LDSR"] = True
										else:
											st.session_state["use_LDSR"] = False

										st.session_state["LDSR_model"] = st.selectbox("LDSR model", st.session_state["LDSR_models"],
																					  index=st.session_state["LDSR_models"].index(st.session_state['defaults'].general.LDSR_model))

										st.session_state["ldsr_sampling_steps"] = int(st.text_input("Sampling Steps", value=st.session_state['defaults'].img2img.LDSR_config.sampling_steps,
																									help=""))

										st.session_state["preDownScale"] = int(st.text_input("PreDownScale", value=st.session_state['defaults'].img2img.LDSR_config.preDownScale,
																							 help=""))

										st.session_state["postDownScale"] = int(st.text_input("postDownScale", value=st.session_state['defaults'].img2img.LDSR_config.postDownScale,
																							  help=""))

										downsample_method_list = ['Nearest', 'Lanczos']
										st.session_state["downsample_method"] = st.selectbox("Downsample Method", downsample_method_list,
																							 index=downsample_method_list.index(st.session_state['defaults'].img2img.LDSR_config.downsample_method))

								else:
									st.session_state["use_LDSR"] = False
									st.session_state["LDSR_model"] = "model"

				with st.expander("Variant"):
					variant_amount = st.slider("Variant Amount:", value=st.session_state['defaults'].img2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
					variant_seed = st.text_input("Variant Seed:", value=st.session_state['defaults'].img2img.variant_seed,
											 help="The seed to use when generating a variant, if left blank a random seed will be generated.")


		with col2_img2img_layout:
			editor_tab = st.tabs(["Editor"])

			editor_image = st.empty()
			st.session_state["editor_image"] = editor_image

			masked_image_holder = st.empty()
			image_holder = st.empty()

			st.form_submit_button("Refresh")

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
				mask_expander.expander("Mask", expanded=True)
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
			with col3_img2img_layout:
				with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders,index=[0]):
					load_models(use_LDSR=st.session_state["use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
								use_GFPGAN=st.session_state["use_GFPGAN"], GFPGAN_model=st.session_state["GFPGAN_model"] ,
								use_RealESRGAN=st.session_state["use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"],
								CustomModel_available=server_state["CustomModel_available"], custom_model=st.session_state["custom_model"])

			if uploaded_images:
				image = Image.open(uploaded_images).convert('RGBA')
				new_img = image.resize((width, height))
				#img_array = np.array(image) # if you want to pass it to OpenCV
				new_mask = None
				if uploaded_masks:
					mask = Image.open(uploaded_masks).convert('RGBA')
					new_mask = mask.resize((width, height))

				try:
					output_images, seed, info, stats = img2img(prompt=prompt, init_info=new_img, init_info_mask=new_mask, mask_mode=mask_mode,
										   mask_restore=img2img_mask_restore, ddim_steps=st.session_state["sampling_steps"],
										   sampler_name=st.session_state["sampler_name"], n_iter=st.session_state["batch_count"],
										   cfg_scale=cfg_scale, denoising_strength=st.session_state["denoising_strength"], variant_seed=variant_seed,
										   seed=seed, noise_mode=noise_mode, find_noise_steps=find_noise_steps, width=width,
										   height=height, variant_amount=variant_amount,
										   ddim_eta=st.session_state.defaults.img2img.ddim_eta, write_info_files=write_info_files,
										   separate_prompts=separate_prompts, normalize_prompt_weights=normalize_prompt_weights,
										   save_individual_images=save_individual_images, save_grid=save_grid,
										   group_by_prompt=group_by_prompt, save_as_jpg=save_as_jpg, use_GFPGAN=st.session_state["use_GFPGAN"],
										   GFPGAN_model=st.session_state["GFPGAN_model"],
										   use_RealESRGAN=st.session_state["use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"],
										   use_LDSR=st.session_state["use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
										   loopback=loopback
										   )

					#show a message when the generation is complete.
					message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")

				except (StopException, KeyError):
					print(f"Received Streamlit StopException")

				# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
				# use the current col2 first tab to show the preview_img and update it as its generated.
				#preview_image.image(output_images, width=750)

#on import run init

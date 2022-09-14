from webui_streamlit import st
from sd_utils import *

from PIL import Image, ImageOps
import torch
import k_diffusion as K
import numpy as np
import time
import torch
import skimage
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

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
	    write_info_files:bool = True, RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B",
	    separate_prompts:bool = False, normalize_prompt_weights:bool = True,
	    save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
	    save_as_jpg: bool = True, use_GFPGAN: bool = True, use_RealESRGAN: bool = True, loopback: bool = False,
	    random_seed_loopback: bool = False
	    ):

	outpath = defaults.general.outdir_img2img or defaults.general.outdir or "outputs/img2img-samples"
	#err = False
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
	
#

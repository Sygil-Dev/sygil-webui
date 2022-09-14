from webui_streamlit import st
from sd_utils import *

from streamlit import StopException

import os
from PIL import Image
import torch
import numpy as np
import time
from torch import autocast

# we use python-slugify to make the filenames safe for windows and linux, its better than doing it manually
# install it with 'pip install python-slugify'
from slugify import slugify

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass

class plugin_info():
        plugname = "txt2img"
        description = "Text to Image"
        isTab = True
        displayPriority = 1
        

if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
	GFPGAN_available = True
else:
	GFPGAN_available = False

if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{defaults.general.RealESRGAN_model}.pth")):
	RealESRGAN_available = True
else:
	RealESRGAN_available = False	

#
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


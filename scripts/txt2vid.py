# base webui import and utils.
from webui_streamlit import st
from sd_utils import *

# streamlit imports
from streamlit import StopException
from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
from streamlit.elements import image as STImage

#other imports

import os
from PIL import Image
import torch
import numpy as np
import time, inspect, timeit
import torch
from torch import autocast
from io import BytesIO
import imageio
from slugify import slugify

# Temp imports

# these are for testing txt2vid, should be removed and we should use things from our own code.
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

# end of imports
#---------------------------------------------------------------------------------------------------------------

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


if os.path.exists(os.path.join(st.session_state['defaults'].general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
	GFPGAN_available = True
else:
	GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments","pretrained_models", f"{st.session_state['defaults'].txt2vid.RealESRGAN_model}.pth")):
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

	if "current_chunk_speed" not in st.session_state:
		st.session_state["current_chunk_speed"] = 0

	if "previous_chunk_speed_list" not in st.session_state:
		st.session_state["previous_chunk_speed_list"] = [0]
		st.session_state["previous_chunk_speed_list"].append(st.session_state["current_chunk_speed"])

	if "update_preview_frequency_list" not in st.session_state:
		st.session_state["update_preview_frequency_list"] = [0]
		st.session_state["update_preview_frequency_list"].append(st.session_state['defaults'].txt2vid.update_preview_frequency)


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
		if st.session_state['defaults'].txt2vid.update_preview:
			step_counter += 1

			if st.session_state['defaults'].txt2vid.update_preview_frequency == step_counter or step_counter == st.session_state.sampling_steps:
				if st.session_state.dynamic_preview_frequency:
					st.session_state["current_chunk_speed"], st.session_state["previous_chunk_speed_list"], st.session_state['defaults'].txt2vid.update_preview_frequency, st.session_state["avg_update_preview_frequency"] = optimize_update_preview_frequency(st.session_state["current_chunk_speed"], st.session_state["previous_chunk_speed_list"], st.session_state['defaults'].txt2vid.update_preview_frequency, st.session_state["update_preview_frequency_list"])   

				#scale and decode the image latents with vae
				cond_latents_2 = 1 / 0.18215 * cond_latents
				image = pipe.vae.decode(cond_latents_2)

				# generate output numpy image as uint8
				image = torch.clamp((image["sample"] + 1.0) / 2.0, min=0.0, max=1.0)
				image = transforms.ToPILImage()(image.squeeze_(0))

				st.session_state["preview_image"].image(image)

				step_counter = 0

		duration = timeit.default_timer() - start

		st.session_state["current_chunk_speed"] = duration

		if duration >= 1:
			speed = "s/it"
		else:
			speed = "it/s"
			duration = 1 / duration

		if i > st.session_state.sampling_steps:
			inference_counter += 1
			inference_percent = int(100 * float(inference_counter + 1 if inference_counter < num_inference_steps else num_inference_steps)/float(num_inference_steps))
			inference_progress = f"{inference_counter + 1 if inference_counter < num_inference_steps else num_inference_steps}/{num_inference_steps} {inference_percent}% "
		else:
			inference_progress = ""

		percent = int(100 * float(i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps)/float(st.session_state.sampling_steps))
		frames_percent = int(100 * float(st.session_state.current_frame if st.session_state.current_frame < st.session_state.max_frames else st.session_state.max_frames)/float(st.session_state.max_frames))

		st.session_state["progress_bar_text"].text(
			f"Running step: {i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps} "
				f"{percent if percent < 100 else 100}% {inference_progress}{duration:.2f}{speed} | "
					f"Frame: {st.session_state.current_frame + 1 if st.session_state.current_frame < st.session_state.max_frames else st.session_state.max_frames}/{st.session_state.max_frames} "
					f"{frames_percent if frames_percent < 100 else 100}% {st.session_state.frame_duration:.2f}{st.session_state.frame_speed}"
		)
		st.session_state["progress_bar"].progress(percent if percent < 100 else 100)

	return image

#
def txt2vid(
	# --------------------------------------
		# args you probably want to change
	prompts = ["blueberry spaghetti", "strawberry spaghetti"], # prompt to dream about
	gpu:int = st.session_state['defaults'].general.gpu, # id of the gpu to run on
	#name:str = 'test', # name of this project, for the output directory
	#rootdir:str = st.session_state['defaults'].general.outdir,
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
				beta_schedule = "scaled_linear",
				starting_image=None
				):
	"""
	prompt = ["blueberry spaghetti", "strawberry spaghetti"], # prompt to dream about
	gpu:int = st.session_state['defaults'].general.gpu, # id of the gpu to run on
	#name:str = 'test', # name of this project, for the output directory
	#rootdir:str = st.session_state['defaults'].general.outdir,
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
	#max_frames +=1

	assert torch.cuda.is_available()
	assert height % 8 == 0 and width % 8 == 0
	torch.manual_seed(seeds)
	torch_device = f"cuda:{gpu}"

	# init the output dir
	sanitized_prompt = slugify(prompts)

	full_path = os.path.join(os.getcwd(), st.session_state['defaults'].general.outdir, "txt2vid-samples", "samples", sanitized_prompt)

	if len(full_path) > 220:
		sanitized_prompt = sanitized_prompt[:220-len(full_path)]
		full_path = os.path.join(os.getcwd(), st.session_state['defaults'].general.outdir, "txt2vid-samples", "samples", sanitized_prompt)

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
	st.session_state["progress_bar_text"].text("Loading models...")	
	
	try:
		if "model" in st.session_state:
			del st.session_state["model"]
	except:
		pass

	#print (st.session_state["weights_path"] != weights_path)

	try:
		if not "pipe" in st.session_state or st.session_state["weights_path"] != weights_path:
			if st.session_state["weights_path"] != weights_path:
				del st.session_state["weights_path"]

			st.session_state["weights_path"] = weights_path
			st.session_state["pipe"] = StableDiffusionPipeline.from_pretrained(
				weights_path,
					use_local_file=True,
							use_auth_token=True,
							torch_dtype=torch.float16 if st.session_state['defaults'].general.use_float16 else None,
										revision="fp16" if not st.session_state['defaults'].general.no_half else None
			)

			st.session_state["pipe"].unet.to(torch_device)
			st.session_state["pipe"].vae.to(torch_device)
			st.session_state["pipe"].text_encoder.to(torch_device)
			
			if st.session_state.defaults.general.enable_attention_slicing:
				st.session_state["pipe"].enable_attention_slicing()
			if st.session_state.defaults.general.enable_minimal_memory_usage:	
				st.session_state["pipe"].enable_minimal_memory_usage()
				
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
			torch_dtype=torch.float16 if st.session_state['defaults'].general.use_float16 else None,
			revision="fp16" if not st.session_state['defaults'].general.no_half else None
		)

		st.session_state["pipe"].unet.to(torch_device)
		st.session_state["pipe"].vae.to(torch_device)
		st.session_state["pipe"].text_encoder.to(torch_device)
		
		if st.session_state.defaults.general.enable_attention_slicing:
			st.session_state["pipe"].enable_attention_slicing()
			
			
		print("Tx2Vid Model Loaded")

	st.session_state["pipe"].scheduler = SCHEDULERS[scheduler]

	# get the conditional text embeddings based on the prompt
	text_input = st.session_state["pipe"].tokenizer(prompts, padding="max_length", max_length=st.session_state["pipe"].tokenizer.model_max_length, truncation=True, return_tensors="pt")
	cond_embeddings = st.session_state["pipe"].text_encoder(text_input.input_ids.to(torch_device))[0] # shape [1, 77, 768]

	#
	if st.session_state.defaults.general.use_sd_concepts_library:

		prompt_tokens = re.findall('<([a-zA-Z0-9-]+)>', prompts)    

		if prompt_tokens:
			# compviz
			#tokenizer = (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelCS).cond_stage_model.tokenizer
			#text_encoder = (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelCS).cond_stage_model.transformer

			# diffusers
			tokenizer = st.session_state.pipe.tokenizer
			text_encoder = st.session_state.pipe.text_encoder

			ext = ('pt', 'bin')
			#print (prompt_tokens)
			
			if len(prompt_tokens) > 1:                                      
				for token_name in prompt_tokens:
					embedding_path = os.path.join(st.session_state['defaults'].general.sd_concepts_library_folder, token_name)	
					if os.path.exists(embedding_path):
						for files in os.listdir(embedding_path):
							if files.endswith(ext):
								load_learned_embed_in_clip(f"{os.path.join(embedding_path, files)}", text_encoder, tokenizer, f"<{token_name}>")
			else:
				embedding_path = os.path.join(st.session_state['defaults'].general.sd_concepts_library_folder, prompt_tokens[0])
				if os.path.exists(embedding_path):
					for files in os.listdir(embedding_path):
						if files.endswith(ext):
							load_learned_embed_in_clip(f"{os.path.join(embedding_path, files)}", text_encoder, tokenizer, f"<{prompt_tokens[0]}>")  	 

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

	st.session_state["total_frames_avg_duration"] = []
	st.session_state["total_frames_avg_speed"] = []

	try:
		while frame_index < max_frames:
			st.session_state["frame_duration"] = 0
			st.session_state["frame_speed"] = 0
			st.session_state["current_frame"] = frame_index

			# sample the destination
			init2 = torch.randn((1, st.session_state["pipe"].unet.in_channels, height // 8, width // 8), device=torch_device)

			for i, t in enumerate(np.linspace(0, 1, max_frames)):
				start = timeit.default_timer()
				print(f"COUNT: {frame_index+1}/{max_frames}")

				#if use_lerp_for_text:
					#init = torch.lerp(init1, init2, float(t))
				#else:
					#init = slerp(gpu, float(t), init1, init2)

				init = slerp(gpu, float(t), init1, init2)

				with autocast("cuda"):
					image = diffuse(st.session_state["pipe"], cond_embeddings, init, num_inference_steps, cfg_scale, eta)

				#im = Image.fromarray(image)
				outpath = os.path.join(full_path, 'frame%06d.png' % frame_index)
				image.save(outpath, quality=quality)

				# send the image to the UI to update it
				#st.session_state["preview_image"].image(im)

				#append the frames to the frames list so we can use them later.
				frames.append(np.asarray(image))

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
		#writer = imageio.get_writer(os.path.join(os.getcwd(), st.session_state['defaults'].general.outdir, "txt2vid-samples"), im, extension=".mp4", fps=30)
		try:
			video_path = os.path.join(os.getcwd(), st.session_state['defaults'].general.outdir, "txt2vid-samples","temp.mp4")
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

	return video_path, seeds, info, stats

#on import run init
def createHTMLGallery(images,info):
	html3 = """
		<div class="gallery-history" style="
	    display: flex;
	    flex-wrap: wrap;
		align-items: flex-start;">
		"""
	mkdwn_array = []
	for i in images:
		try:
			seed = info[images.index(i)]
		except:
			seed = ' '
		image_io = BytesIO()
		i.save(image_io, 'PNG')
		width, height = i.size
		#get random number for the id
		image_id = "%s" % (str(images.index(i)))
		(data, mimetype) = STImage._normalize_to_bytes(image_io.getvalue(), width, 'auto')
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
			<div class="desc" style="text-align: center; opacity: 40%;">{seed}</div>
			</div>
			'''
		mkdwn_array.append(mkdwn)

	html3 += "".join(mkdwn_array)
	html3 += '</div>'
	return html3
#
def layout():
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
			width = st.slider("Width:", min_value=64, max_value=2048, value=st.session_state['defaults'].txt2vid.width, step=64)
			height = st.slider("Height:", min_value=64, max_value=2048, value=st.session_state['defaults'].txt2vid.height, step=64)
			cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=st.session_state['defaults'].txt2vid.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")

			#uploaded_images = st.file_uploader("Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "webp"],
												#help="Upload an image which will be used for the image to image generation.")			
			seed = st.text_input("Seed:", value=st.session_state['defaults'].txt2vid.seed, help=" The seed to use, if left blank a random seed will be generated.")
			#batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=st.session_state['defaults'].txt2vid.batch_count, step=1, help="How many iterations or batches of images to generate in total.")
			#batch_size = st.slider("Batch size", min_value=1, max_value=250, value=st.session_state['defaults'].txt2vid.batch_size, step=1,
					#help="How many images are at once in a batch.\
					#It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
					#Default: 1")

			st.session_state["max_frames"] = int(st.text_input("Max Frames:", value=st.session_state['defaults'].txt2vid.max_frames, help="Specify the max number of frames you want to generate."))

			with st.expander("Preview Settings"):
				st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=st.session_state['defaults'].txt2vid.update_preview,
																 help="If enabled the image preview will be updated during the generation instead of at the end. \
					                                         You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
					                                         By default this is enabled and the frequency is set to 1 step.")

				st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=st.session_state['defaults'].txt2vid.update_preview_frequency,
																			 help="Frequency in steps at which the the preview image is updated. By default the frequency \
																			 is set to 1 step.")
				
			#
			
			
			
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

			#generate_video = st.empty()
			st.session_state["preview_video"] = st.empty()

			message = st.empty()

		with gallery_tab:
			st.write('Here should be the image gallery, if I could make a grid in streamlit.')

	with col3:
		# If we have custom models available on the "models/custom"
		#folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
		if st.session_state["CustomModel_available"]:
			custom_model = st.selectbox("Custom Model:", st.session_state["defaults"].txt2vid.custom_models_list,
										index=st.session_state["defaults"].txt2vid.custom_models_list.index(st.session_state["defaults"].txt2vid.default_model),
											help="Select the model you want to use. This option is only available if you have custom models \
				                            on your 'models/custom' folder. The model name that will be shown here is the same as the name\
				                            the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
				                        will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
		else:
			custom_model = "CompVis/stable-diffusion-v1-4"

		#st.session_state["weights_path"] = custom_model
		#else:
			#custom_model = "CompVis/stable-diffusion-v1-4"
			#st.session_state["weights_path"] = f"CompVis/{slugify(custom_model.lower())}"

		st.session_state.sampling_steps = st.slider("Sampling Steps",
		value=st.session_state['defaults'].txt2vid.sampling_steps,
		min_value=st.session_state['defaults'].txt2vid.slider_bounds.sampling.lower,
		max_value=st.session_state['defaults'].txt2vid.slider_bounds.sampling.upper,
		step=st.session_state['defaults'].txt2vid.slider_steps.sampling,
		help="Number of steps between each pair of sampled points")
		st.session_state.num_inference_steps = st.slider("Inference Steps:", value=st.session_state['defaults'].txt2vid.num_inference_steps, min_value=10,step=10, max_value=500,
														 help="Higher values (e.g. 100, 200 etc) can create better images.")

		#sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
		#sampler_name = st.selectbox("Sampling method", sampler_name_list,
						#index=sampler_name_list.index(st.session_state['defaults'].txt2vid.default_sampler), help="Sampling method to use. Default: k_euler")
		scheduler_name_list = ["klms", "ddim"]
		scheduler_name = st.selectbox("Scheduler:", scheduler_name_list,
									  index=scheduler_name_list.index(st.session_state['defaults'].txt2vid.scheduler_name), help="Scheduler to use. Default: klms")

		beta_scheduler_type_list = ["scaled_linear", "linear"]
		beta_scheduler_type = st.selectbox("Beta Schedule Type:", beta_scheduler_type_list,
										   index=beta_scheduler_type_list.index(st.session_state['defaults'].txt2vid.beta_scheduler_type), help="Schedule Type to use. Default: linear")


		#basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

		#with basic_tab:
			#summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
				#help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")

		with st.expander("Advanced"):
			st.session_state["separate_prompts"] = st.checkbox("Create Prompt Matrix.", value=st.session_state['defaults'].txt2vid.separate_prompts,
															   help="Separate multiple prompts using the `|` character, and get all combinations of them.")
			st.session_state["normalize_prompt_weights"] = st.checkbox("Normalize Prompt Weights.",
																	   value=st.session_state['defaults'].txt2vid.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
			st.session_state["save_individual_images"] = st.checkbox("Save individual images.",
																	 value=st.session_state['defaults'].txt2vid.save_individual_images, help="Save each image generated before any filter or enhancement is applied.")
			st.session_state["save_video"] = st.checkbox("Save video",value=st.session_state['defaults'].txt2vid.save_video, help="Save a video with all the images generated as frames at the end of the generation.")
			st.session_state["group_by_prompt"] = st.checkbox("Group results by prompt", value=st.session_state['defaults'].txt2vid.group_by_prompt,
															  help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
			st.session_state["write_info_files"] = st.checkbox("Write Info file", value=st.session_state['defaults'].txt2vid.write_info_files,
															   help="Save a file next to the image with informartion about the generation.")
			st.session_state["dynamic_preview_frequency"] = st.checkbox("Dynamic Preview Frequency", value=st.session_state['defaults'].txt2vid.dynamic_preview_frequency,
																		help="This option tries to find the best value at which we can update \
					                                           the preview image during generation while minimizing the impact it has in performance. Default: True")
			st.session_state["do_loop"] = st.checkbox("Do Loop", value=st.session_state['defaults'].txt2vid.do_loop,
													  help="Do loop")
			st.session_state["save_as_jpg"] = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].txt2vid.save_as_jpg, help="Saves the images as jpg instead of png.")

			if GFPGAN_available:
				st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].txt2vid.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation. This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
			else:
				st.session_state["use_GFPGAN"] = False

			if RealESRGAN_available:
				st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=st.session_state['defaults'].txt2vid.use_RealESRGAN,
																 help="Uses the RealESRGAN model to upscale the images after the generation. This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
				st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)
			else:
				st.session_state["use_RealESRGAN"] = False
				st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"

			st.session_state["variant_amount"] = st.slider("Variant Amount:", value=st.session_state['defaults'].txt2vid.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
			st.session_state["variant_seed"] = st.text_input("Variant Seed:", value=st.session_state['defaults'].txt2vid.seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")
			st.session_state["beta_start"] = st.slider("Beta Start:", value=st.session_state['defaults'].txt2vid.beta_start, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")
			st.session_state["beta_end"] = st.slider("Beta End:", value=st.session_state['defaults'].txt2vid.beta_end, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")

	if generate_button:
		#print("Loading models")
		# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.
		#load_models(False, False, False, st.session_state["RealESRGAN_model"], CustomModel_available=st.session_state["CustomModel_available"], custom_model=custom_model)

		try:
			# run video generation
			video, seed, info, stats = txt2vid(prompts=prompt, gpu=st.session_state["defaults"].general.gpu,
											   num_steps=st.session_state.sampling_steps, max_frames=int(st.session_state.max_frames),
							   num_inference_steps=st.session_state.num_inference_steps,
							   cfg_scale=cfg_scale,do_loop=st.session_state["do_loop"],
							   seeds=seed, quality=100, eta=0.0, width=width,
							   height=height, weights_path=custom_model, scheduler=scheduler_name,
							   disable_tqdm=False, beta_start=st.session_state["beta_start"], beta_end=st.session_state["beta_end"],
							   beta_schedule=beta_scheduler_type, starting_image=None)

			#message.success('Done!', icon="✅")
			message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")

			#history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont = st.session_state['historyTab']

			#if 'latestVideos' in st.session_state:
				#for i in video:
					##push the new image to the list of latest images and remove the oldest one
					##remove the last index from the list\
					#st.session_state['latestVideos'].pop()
					##add the new image to the start of the list
					#st.session_state['latestVideos'].insert(0, i)
				#PlaceHolder.empty()

				#with PlaceHolder.container():
					#col1, col2, col3 = st.columns(3)
					#col1_cont = st.container()
					#col2_cont = st.container()
					#col3_cont = st.container()

					#with col1_cont:
						#with col1:
							#st.image(st.session_state['latestVideos'][0])
							#st.image(st.session_state['latestVideos'][3])
							#st.image(st.session_state['latestVideos'][6])
					#with col2_cont:
						#with col2:
							#st.image(st.session_state['latestVideos'][1])
							#st.image(st.session_state['latestVideos'][4])
							#st.image(st.session_state['latestVideos'][7])
					#with col3_cont:
						#with col3:
							#st.image(st.session_state['latestVideos'][2])
							#st.image(st.session_state['latestVideos'][5])
							#st.image(st.session_state['latestVideos'][8])
					#historyGallery = st.empty()

				## check if output_images length is the same as seeds length
				#with gallery_tab:
					#st.markdown(createHTMLGallery(video,seed), unsafe_allow_html=True)


				#st.session_state['historyTab'] = [history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont]

		except (StopException, KeyError):
			print(f"Received Streamlit StopException")



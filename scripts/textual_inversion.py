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
from sd_utils import st, set_page_title, seed_to_int

# streamlit imports
from streamlit.runtime.scriptrunner import StopException
from streamlit_tensorboard import st_tensorboard

#streamlit components section
from streamlit_server_state import server_state

#other imports
from transformers import CLIPTextModel, CLIPTokenizer

# Temp imports

import itertools
import math
import os
import random
#import datetime
#from pathlib import Path
#from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel#, PNDMScheduler
from diffusers.optimization import get_scheduler
#from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from pipelines.stable_diffusion.no_check import NoCheck
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from slugify import slugify
import json
import os#, subprocess
#from io import StringIO


# end of imports
#---------------------------------------------------------------------------------------------------------------

logger = get_logger(__name__)

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class TextualInversionDataset(Dataset):
	def __init__(
	    self,
	    data_root,
	    tokenizer,
	    learnable_property="object",  # [object, style]
	    size=512,
	    repeats=100,
	    interpolation="bicubic",
	    set="train",
	    placeholder_token="*",
	    center_crop=False,
	    templates=None
	    ):

		self.data_root = data_root
		self.tokenizer = tokenizer
		self.learnable_property = learnable_property
		self.size = size
		self.placeholder_token = placeholder_token
		self.center_crop = center_crop

		self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if file_path.lower().endswith(('.png', '.jpg', '.jpeg'))]

		self.num_images = len(self.image_paths)
		self._length = self.num_images

		if set == "train":
			self._length = self.num_images * repeats

		self.interpolation = {
		    "linear": PIL.Image.LINEAR,
		    "bilinear": PIL.Image.Resampling.BILINEAR,
		    "bicubic": PIL.Image.Resampling.BICUBIC,
		    "lanczos": PIL.Image.Resampling.LANCZOS,
		    }[interpolation]

		self.templates = templates
		self.cache = {}
		self.tokenized_templates = [self.tokenizer(
		    text.format(self.placeholder_token),
		        padding="max_length",
		        truncation=True,
		        max_length=self.tokenizer.model_max_length,
		        return_tensors="pt",
		        ).input_ids[0] for text in self.templates]

	def __len__(self):
		return self._length

	def get_example(self, image_path, flipped):
		if image_path in self.cache:
			return self.cache[image_path]

		example = {}
		image = Image.open(image_path)

		if not image.mode == "RGB":
			image = image.convert("RGB")

		# default to score-sde preprocessing
		img = np.array(image).astype(np.uint8)
		if self.center_crop:
			crop = min(img.shape[0], img.shape[1])
			h, w, = (
			    img.shape[0],
			    img.shape[1],
			)
			img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
		image = Image.fromarray(img)
		image = image.resize((self.size, self.size), resample=self.interpolation)
		image = transforms.RandomHorizontalFlip(p=1 if flipped else 0)(image)
		image = np.array(image).astype(np.uint8)
		image = (image / 127.5 - 1.0).astype(np.float32)
		example["key"] = "-".join([image_path, "-", str(flipped)])
		example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

		self.cache[image_path] = example
		return example

	def __getitem__(self, i):
		flipped = random.choice([False, True])
		example = self.get_example(self.image_paths[i % self.num_images], flipped)
		example["input_ids"] = random.choice(self.tokenized_templates)
		return example


def freeze_params(params):
	for param in params:
		param.requires_grad = False


def save_resume_file(basepath, extra = {}, config=''):
	info = {"args": config["args"]}
	info["args"].update(extra)

	with open(f"{os.path.join(basepath, 'resume.json')}", "w") as f:
		#print (info)
		json.dump(info, f, indent=4)

	with open(f"{basepath}/token_identifier.txt", "w") as f:
		f.write(f"{config['args']['placeholder_token']}")

	with open(f"{basepath}/type_of_concept.txt", "w") as f:
		f.write(f"{config['args']['learnable_property']}")

	config['args'] = info["args"]

	return config['args']

class Checkpointer:
	def __init__(
	    self,
	    accelerator,
	    vae,
	    unet,
	    tokenizer,
	    placeholder_token,
	    placeholder_token_id,
	    templates,
	    output_dir,
	    random_sample_batches,
	    sample_batch_size,
	    stable_sample_batches,
	    seed
	    ):
		self.accelerator = accelerator
		self.vae = vae
		self.unet = unet
		self.tokenizer = tokenizer
		self.placeholder_token = placeholder_token
		self.placeholder_token_id = placeholder_token_id
		self.templates = templates
		self.output_dir = output_dir
		self.seed = seed
		self.random_sample_batches = random_sample_batches
		self.sample_batch_size = sample_batch_size
		self.stable_sample_batches = stable_sample_batches

	@torch.no_grad()
	def checkpoint(self, step, text_encoder, save_samples=True, path=None):
		print("Saving checkpoint for step %d..." % step)
		with torch.autocast("cuda"):
			if path is None:
				checkpoints_path = f"{self.output_dir}/checkpoints"
				os.makedirs(checkpoints_path, exist_ok=True)

			unwrapped = self.accelerator.unwrap_model(text_encoder)

			# Save a checkpoint
			learned_embeds = unwrapped.get_input_embeddings().weight[self.placeholder_token_id]
			learned_embeds_dict = {self.placeholder_token: learned_embeds.detach().cpu()}

			filename = f"%s_%d.bin" % (slugify(self.placeholder_token), step)
			if path is not None:
				torch.save(learned_embeds_dict, path)
			else:
				torch.save(learned_embeds_dict, f"{checkpoints_path}/{filename}")
				torch.save(learned_embeds_dict, f"{checkpoints_path}/last.bin")

			del unwrapped
			del learned_embeds


	@torch.no_grad()
	def save_samples(self, step, text_encoder, height, width, guidance_scale, eta, num_inference_steps):
		samples_path = f"{self.output_dir}/concept_images"
		os.makedirs(samples_path, exist_ok=True)

		#if "checker" not in server_state['textual_inversion']:
		#with server_state_lock['textual_inversion']["checker"]:
		server_state['textual_inversion']["checker"] = NoCheck()

		#if "unwrapped" not in server_state['textual_inversion']:
		#	with server_state_lock['textual_inversion']["unwrapped"]:
		server_state['textual_inversion']["unwrapped"] = self.accelerator.unwrap_model(text_encoder)

		#if "pipeline" not in server_state['textual_inversion']:
		#	with server_state_lock['textual_inversion']["pipeline"]:
		# Save a sample image
		server_state['textual_inversion']["pipeline"] = StableDiffusionPipeline(
			text_encoder=server_state['textual_inversion']["unwrapped"],
			vae=self.vae,
			unet=self.unet,
			tokenizer=self.tokenizer,
			scheduler=LMSDiscreteScheduler(
				beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
				),
			safety_checker=NoCheck(),
			feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
			).to("cuda")

		server_state['textual_inversion']["pipeline"].enable_attention_slicing()

		if self.stable_sample_batches > 0:
			stable_latents = torch.randn(
			    (self.sample_batch_size, server_state['textual_inversion']["pipeline"].unet.in_channels, height // 8, width // 8),
			    device=server_state['textual_inversion']["pipeline"].device,
			    generator=torch.Generator(device=server_state['textual_inversion']["pipeline"].device).manual_seed(self.seed),
			)

			stable_prompts = [choice.format(self.placeholder_token) for choice in (self.templates * self.sample_batch_size)[:self.sample_batch_size]]

			# Generate and save stable samples
			for i in range(0, self.stable_sample_batches):
				samples = server_state['textual_inversion']["pipeline"](
				    prompt=stable_prompts,
				    height=384,
				    latents=stable_latents,
				    width=384,
				    guidance_scale=guidance_scale,
				    eta=eta,
				    num_inference_steps=num_inference_steps,
				    output_type='pil'
				    )["sample"]

				for idx, im in enumerate(samples):
					filename = f"stable_sample_%d_%d_step_%d.png" % (i+1, idx+1, step)
					im.save(f"{samples_path}/{filename}")
				del samples
			del stable_latents

		prompts = [choice.format(self.placeholder_token) for choice in random.choices(self.templates, k=self.sample_batch_size)]
		# Generate and save random samples
		for i in range(0, self.random_sample_batches):
			samples = server_state['textual_inversion']["pipeline"](
			    prompt=prompts,
			    height=384,
			    width=384,
			    guidance_scale=guidance_scale,
			    eta=eta,
			    num_inference_steps=num_inference_steps,
			    output_type='pil'
			    )["sample"]
			for idx, im in enumerate(samples):
				filename = f"step_%d_sample_%d_%d.png" % (step, i+1, idx+1)
				im.save(f"{samples_path}/{filename}")
			del samples

		del server_state['textual_inversion']["checker"]
		del server_state['textual_inversion']["unwrapped"]
		del server_state['textual_inversion']["pipeline"]
		torch.cuda.empty_cache()

#@retry(RuntimeError, tries=5)
def textual_inversion(config):
	print ("Running textual inversion.")

	#if "pipeline" in server_state["textual_inversion"]:
		#del server_state['textual_inversion']["checker"]
		#del server_state['textual_inversion']["unwrapped"]
		#del server_state['textual_inversion']["pipeline"]
		#torch.cuda.empty_cache()

	global_step_offset = 0

	#print(config['args']['resume_from'])
	if config['args']['resume_from']:
		try:
			basepath = f"{config['args']['resume_from']}"

			with open(f"{basepath}/resume.json", 'r') as f:
				state = json.load(f)

			global_step_offset = state["args"].get("global_step", 0)

			print("Resuming state from %s" % config['args']['resume_from'])
			print("We've trained %d steps so far" % global_step_offset)

		except json.decoder.JSONDecodeError:
			pass
	else:
		basepath = f"{config['args']['output_dir']}/{slugify(config['args']['placeholder_token'])}"
		os.makedirs(basepath, exist_ok=True)


	accelerator = Accelerator(
	    gradient_accumulation_steps=config['args']['gradient_accumulation_steps'],
	    mixed_precision=config['args']['mixed_precision']
	)

	# If passed along, set the training seed.
	if config['args']['seed']:
		set_seed(config['args']['seed'])

	#if "tokenizer" not in server_state["textual_inversion"]:
	# Load the tokenizer and add the placeholder token as a additional special token
	#with server_state_lock['textual_inversion']["tokenizer"]:
	if config['args']['tokenizer_name']:
		server_state['textual_inversion']["tokenizer"] = CLIPTokenizer.from_pretrained(config['args']['tokenizer_name'])
	elif config['args']['pretrained_model_name_or_path']:
		server_state['textual_inversion']["tokenizer"] = CLIPTokenizer.from_pretrained(
	        config['args']['pretrained_model_name_or_path'] + '/tokenizer'
	    )

	# Add the placeholder token in tokenizer
	num_added_tokens = server_state['textual_inversion']["tokenizer"].add_tokens(config['args']['placeholder_token'])
	if num_added_tokens == 0:
		st.error(
		    f"The tokenizer already contains the token {config['args']['placeholder_token']}. Please pass a different"
		    " `placeholder_token` that is not already in the tokenizer."
		)

	# Convert the initializer_token, placeholder_token to ids
	token_ids = server_state['textual_inversion']["tokenizer"].encode(config['args']['initializer_token'], add_special_tokens=False)
	# Check if initializer_token is a single token or a sequence of tokens
	if len(token_ids) > 1:
		st.error("The initializer token must be a single token.")

	initializer_token_id = token_ids[0]
	placeholder_token_id = server_state['textual_inversion']["tokenizer"].convert_tokens_to_ids(config['args']['placeholder_token'])

	#if "text_encoder" not in server_state['textual_inversion']:
	# Load models and create wrapper for stable diffusion
	#with server_state_lock['textual_inversion']["text_encoder"]:
	server_state['textual_inversion']["text_encoder"] = CLIPTextModel.from_pretrained(
        config['args']['pretrained_model_name_or_path'] + '/text_encoder',
        )

	#if "vae" not in server_state['textual_inversion']:
		#with server_state_lock['textual_inversion']["vae"]:
	server_state['textual_inversion']["vae"] = AutoencoderKL.from_pretrained(
        config['args']['pretrained_model_name_or_path'] + '/vae',
    )

	#if "unet" not in server_state['textual_inversion']:
		#with server_state_lock['textual_inversion']["unet"]:
	server_state['textual_inversion']["unet"] = UNet2DConditionModel.from_pretrained(
        config['args']['pretrained_model_name_or_path'] + '/unet',
    )

	base_templates = imagenet_style_templates_small if config['args']['learnable_property'] == "style" else imagenet_templates_small
	if config['args']['custom_templates']:
		templates = config['args']['custom_templates'].split(";")
	else:
		templates = base_templates

	slice_size = server_state['textual_inversion']["unet"].config.attention_head_dim // 2
	server_state['textual_inversion']["unet"].set_attention_slice(slice_size)

	# Resize the token embeddings as we are adding new special tokens to the tokenizer
	server_state['textual_inversion']["text_encoder"].resize_token_embeddings(len(server_state['textual_inversion']["tokenizer"]))

	# Initialise the newly added placeholder token with the embeddings of the initializer token
	token_embeds = server_state['textual_inversion']["text_encoder"].get_input_embeddings().weight.data

	if "resume_checkpoint" in config['args']:
		if config['args']['resume_checkpoint'] is not None:
			token_embeds[placeholder_token_id] = torch.load(config['args']['resume_checkpoint'])[config['args']['placeholder_token']]
	else:
		token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

	# Freeze vae and unet
	freeze_params(server_state['textual_inversion']["vae"].parameters())
	freeze_params(server_state['textual_inversion']["unet"].parameters())
	# Freeze all parameters except for the token embeddings in text encoder
	params_to_freeze = itertools.chain(
	    server_state['textual_inversion']["text_encoder"].text_model.encoder.parameters(),
	    server_state['textual_inversion']["text_encoder"].text_model.final_layer_norm.parameters(),
	    server_state['textual_inversion']["text_encoder"].text_model.embeddings.position_embedding.parameters(),
	)
	freeze_params(params_to_freeze)

	checkpointer = Checkpointer(
	    accelerator=accelerator,
	    vae=server_state['textual_inversion']["vae"],
	    unet=server_state['textual_inversion']["unet"],
	    tokenizer=server_state['textual_inversion']["tokenizer"],
	    placeholder_token=config['args']['placeholder_token'],
	    placeholder_token_id=placeholder_token_id,
	    templates=templates,
	    output_dir=basepath,
	    sample_batch_size=config['args']['sample_batch_size'],
	    random_sample_batches=config['args']['random_sample_batches'],
	    stable_sample_batches=config['args']['stable_sample_batches'],
	    seed=config['args']['seed']
	)

	if config['args']['scale_lr']:
		config['args']['learning_rate'] = (
		    config['args']['learning_rate'] * config[
				'args']['gradient_accumulation_steps'] * config['args']['train_batch_size'] * accelerator.num_processes
		)

	# Initialize the optimizer
	optimizer = torch.optim.AdamW(
	    server_state['textual_inversion']["text_encoder"].get_input_embeddings().parameters(),  # only optimize the embeddings
	    lr=config['args']['learning_rate'],
	    betas=(config['args']['adam_beta1'], config['args']['adam_beta2']),
	    weight_decay=config['args']['adam_weight_decay'],
	    eps=config['args']['adam_epsilon'],
	)

	# TODO (patil-suraj): load scheduler using args
	noise_scheduler = DDPMScheduler(
	    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
	)

	train_dataset = TextualInversionDataset(
	    data_root=config['args']['train_data_dir'],
	    tokenizer=server_state['textual_inversion']["tokenizer"],
	    size=config['args']['resolution'],
	    placeholder_token=config['args']['placeholder_token'],
	    repeats=config['args']['repeats'],
	    learnable_property=config['args']['learnable_property'],
	    center_crop=config['args']['center_crop'],
	    set="train",
	    templates=templates
	)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['args']['train_batch_size'], shuffle=True)

	# Scheduler and math around the number of training steps.
	overrode_max_train_steps = False
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['args']['gradient_accumulation_steps'])
	if config['args']['max_train_steps'] is None:
		config['args']['max_train_steps'] = config['args']['num_train_epochs'] * num_update_steps_per_epoch
		overrode_max_train_steps = True

	lr_scheduler = get_scheduler(
	    config['args']['lr_scheduler'],
	    optimizer=optimizer,
	    num_warmup_steps=config['args']['lr_warmup_steps'] * config['args']['gradient_accumulation_steps'],
	    num_training_steps=config['args']['max_train_steps'] * config['args']['gradient_accumulation_steps'],
	)

	server_state['textual_inversion']["text_encoder"], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
	    server_state['textual_inversion']["text_encoder"], optimizer, train_dataloader, lr_scheduler
	)

	# Move vae and unet to device
	server_state['textual_inversion']["vae"].to(accelerator.device)
	server_state['textual_inversion']["unet"].to(accelerator.device)

	# Keep vae and unet in eval mode as we don't train these
	server_state['textual_inversion']["vae"].eval()
	server_state['textual_inversion']["unet"].eval()

	# We need to recalculate our total training steps as the size of the training dataloader may have changed.
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['args']['gradient_accumulation_steps'])
	if overrode_max_train_steps:
		config['args']['max_train_steps'] = config['args']['num_train_epochs'] * num_update_steps_per_epoch
	# Afterwards we recalculate our number of training epochs
	config['args']['num_train_epochs'] = math.ceil(config['args']['max_train_steps'] / num_update_steps_per_epoch)

	# We need to initialize the trackers we use, and also store our configuration.
	# The trackers initializes automatically on the main process.
	if accelerator.is_main_process:
		accelerator.init_trackers("textual_inversion", config=config['args'])

	# Train!
	total_batch_size = config['args']['train_batch_size'] * accelerator.num_processes * st.session_state[
		'textual_inversion']['args']['gradient_accumulation_steps']

	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num Epochs = {config['args']['num_train_epochs']}")
	logger.info(f"  Instantaneous batch size per device = {config['args']['train_batch_size']}")
	logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {config['args']['gradient_accumulation_steps']}")
	logger.info(f"  Total optimization steps = {config['args']['max_train_steps']}")
	# Only show the progress bar once on each machine.
	progress_bar = tqdm(range(config['args']['max_train_steps']), disable=not accelerator.is_local_main_process)
	progress_bar.set_description("Steps")
	global_step = 0
	encoded_pixel_values_cache = {}

	try:
		for epoch in range(config['args']['num_train_epochs']):
			server_state['textual_inversion']["text_encoder"].train()
			for step, batch in enumerate(train_dataloader):
				with accelerator.accumulate(server_state['textual_inversion']["text_encoder"]):
					# Convert images to latent space
					key = "|".join(batch["key"])
					if encoded_pixel_values_cache.get(key, None) is None:
						encoded_pixel_values_cache[key] = server_state['textual_inversion']["vae"].encode(batch["pixel_values"]).latent_dist
					latents = encoded_pixel_values_cache[key].sample().detach().half() * 0.18215

					# Sample noise that we'll add to the latents
					noise = torch.randn(latents.shape).to(latents.device)
					bsz = latents.shape[0]
					# Sample a random timestep for each image
					timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

					# Add noise to the latents according to the noise magnitude at each timestep
					# (this is the forward diffusion process)
					noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

					# Get the text embedding for conditioning
					encoder_hidden_states = server_state['textual_inversion']["text_encoder"](batch["input_ids"])[0]

					# Predict the noise residual
					noise_pred = server_state['textual_inversion']["unet"](noisy_latents, timesteps, encoder_hidden_states).sample

					loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
					accelerator.backward(loss)

					# Zero out the gradients for all token embeddings except the newly added
					# embeddings for the concept, as we only want to optimize the concept embeddings
					if accelerator.num_processes > 1:
						grads = server_state['textual_inversion']["text_encoder"].module.get_input_embeddings().weight.grad
					else:
						grads = server_state['textual_inversion']["text_encoder"].get_input_embeddings().weight.grad
					# Get the index for tokens that we want to zero the grads for
					index_grads_to_zero = torch.arange(len(server_state['textual_inversion']["tokenizer"])) != placeholder_token_id
					grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

					optimizer.step()
					lr_scheduler.step()
					optimizer.zero_grad()

				#try:
				# Checks if the accelerator has performed an optimization step behind the scenes
				if accelerator.sync_gradients:
					progress_bar.update(1)
					global_step += 1

					if global_step % config['args']['checkpoint_frequency'] == 0 and global_step > 0 and accelerator.is_main_process:
						checkpointer.checkpoint(global_step + global_step_offset, server_state['textual_inversion']["text_encoder"])
						save_resume_file(basepath, {
					        "global_step": global_step + global_step_offset,
					        "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
					    }, config)

						checkpointer.save_samples(
					        global_step + global_step_offset,
					        server_state['textual_inversion']["text_encoder"],
					        config['args']['resolution'], config['args'][
					            'resolution'], 7.5, 0.0, config['args']['sample_steps'])

						checkpointer.checkpoint(
					        global_step + global_step_offset,
					        server_state['textual_inversion']["text_encoder"],
					        path=f"{basepath}/learned_embeds.bin"
					    )
				#except KeyError:
					#raise StopException

				logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
				progress_bar.set_postfix(**logs)

				#accelerator.log(logs, step=global_step)

				#try:
				if global_step >= config['args']['max_train_steps']:
					break
				#except:
					#pass

			accelerator.wait_for_everyone()

		# Create the pipeline using the trained modules and save it.
		if accelerator.is_main_process:
			print("Finished! Saving final checkpoint and resume state.")
			checkpointer.checkpoint(
			    global_step + global_step_offset,
			    server_state['textual_inversion']["text_encoder"],
			    path=f"{basepath}/learned_embeds.bin"
			)

			save_resume_file(basepath, {
			    "global_step": global_step + global_step_offset,
			    "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
			}, config)

			accelerator.end_training()

	except (KeyboardInterrupt, StopException) as e:
		print(f"Received Streamlit StopException or KeyboardInterrupt")

		if accelerator.is_main_process:
			print("Interrupted, saving checkpoint and resume state...")
			checkpointer.checkpoint(global_step + global_step_offset, server_state['textual_inversion']["text_encoder"])

			config['args'] = save_resume_file(basepath, {
			    "global_step": global_step + global_step_offset,
			    "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
			    }, config)


			checkpointer.checkpoint(
			    global_step + global_step_offset,
			    server_state['textual_inversion']["text_encoder"],
			    path=f"{basepath}/learned_embeds.bin"
			)

		quit()


def layout():

	with st.form("textual-inversion"):
		#st.info("Under Construction. :construction_worker:")
		#parser = argparse.ArgumentParser(description="Simple example of a training script.")

		set_page_title("Textual Inversion - Stable Diffusion Playground")

		config_tab, output_tab, tensorboard_tab = st.tabs(["Textual Inversion Config", "Ouput", "TensorBoard"])

		with config_tab:
			col1, col2, col3, col4, col5 = st.columns(5, gap='large')

			if "textual_inversion" not in st.session_state:
				st.session_state["textual_inversion"] = {}

			if "textual_inversion" not in server_state:
				server_state["textual_inversion"] = {}

			if "args" not in st.session_state["textual_inversion"]:
				st.session_state["textual_inversion"]["args"] = {}


			with col1:
				st.session_state["textual_inversion"]["args"]["pretrained_model_name_or_path"] = st.text_input("Pretrained Model Path",
																									  value=st.session_state["defaults"].textual_inversion.pretrained_model_name_or_path,
																									  help="Path to pretrained model or model identifier from huggingface.co/models.")

				st.session_state["textual_inversion"]["args"]["tokenizer_name"] = st.text_input("Tokenizer Name",
																					   value=st.session_state["defaults"].textual_inversion.tokenizer_name,
																					   help="Pretrained tokenizer name or path if not the same as model_name")

				st.session_state["textual_inversion"]["args"]["train_data_dir"] = st.text_input("train_data_dir", value="", help="A folder containing the training data.")

				st.session_state["textual_inversion"]["args"]["placeholder_token"] = st.text_input("Placeholder Token", value="", help="A token to use as a placeholder for the concept.")

				st.session_state["textual_inversion"]["args"]["initializer_token"] = st.text_input("Initializer Token", value="", help="A token to use as initializer word.")

				st.session_state["textual_inversion"]["args"]["learnable_property"] = st.selectbox("Learnable Property", ["object", "style"], index=0, help="Choose between 'object' and 'style'")

				st.session_state["textual_inversion"]["args"]["repeats"] = int(st.text_input("Number of times to Repeat", value=100, help="How many times to repeat the training data."))

				with col2:
					st.session_state["textual_inversion"]["args"]["output_dir"] = st.text_input("Output Directory",
																		   value=str(os.path.join("outputs", "textual_inversion")),
																		   help="The output directory where the model predictions and checkpoints will be written.")

					st.session_state["textual_inversion"]["args"]["seed"] = seed_to_int(st.text_input("Seed", value=0,
					                                                                                help="A seed for reproducible training, if left empty a random one will be generated. Default: 0"))

					st.session_state["textual_inversion"]["args"]["resolution"] = int(st.text_input("Resolution",  value=512,
																   help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"))

					st.session_state["textual_inversion"]["args"]["center_crop"] = st.checkbox("Center Image", value=True, help="Whether to center crop images before resizing to resolution")

					st.session_state["textual_inversion"]["args"]["train_batch_size"] = int(st.text_input("Train Batch Size",  value=1, help="Batch size (per device) for the training dataloader."))

					st.session_state["textual_inversion"]["args"]["num_train_epochs"] = int(st.text_input("Number of Steps to Train",  value=100, help="Number of steps to train."))

					st.session_state["textual_inversion"]["args"]["max_train_steps"] = int(st.text_input("Max Number of Steps to Train", value=5000,
																	help="Total number of training steps to perform.  If provided, overrides 'Number of Steps to Train'."))

					with col3:
						st.session_state["textual_inversion"]["args"]["gradient_accumulation_steps"] = int(st.text_input("Gradient Accumulation Steps",  value=1,
																						help="Number of updates steps to accumulate before performing a backward/update pass."))

						st.session_state["textual_inversion"]["args"]["learning_rate"] = float(st.text_input("Learning Rate", value=5.0e-04,
																		  help="Initial learning rate (after the potential warmup period) to use."))

						st.session_state["textual_inversion"]["args"]["scale_lr"] = st.checkbox("Scale Learning Rate", value=True,
																   help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")

						st.session_state["textual_inversion"]["args"]["lr_scheduler"] = st.text_input("Learning Rate Scheduler",  value="constant",
																		 help=("The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
																			   " 'constant', 'constant_with_warmup']" ))

						st.session_state["textual_inversion"]["args"]["lr_warmup_steps"] = int(st.text_input("Learning Rate Warmup Steps", value=500, help="Number of steps for the warmup in the lr scheduler."))

						st.session_state["textual_inversion"]["args"]["adam_beta1"] = float(st.text_input("Adam Beta 1",  value=0.9, help="The beta1 parameter for the Adam optimizer."))

						st.session_state["textual_inversion"]["args"]["adam_beta2"] = float(st.text_input("Adam Beta 2", value=0.999, help="The beta2 parameter for the Adam optimizer."))

						st.session_state["textual_inversion"]["args"]["adam_weight_decay"] = float(st.text_input("Adam Weight Decay",  value=1e-2, help="Weight decay to use."))

						st.session_state["textual_inversion"]["args"]["adam_epsilon"] = float(st.text_input("Adam Epsilon",  value=1e-08, help="Epsilon value for the Adam optimizer"))

						with col4:
							st.session_state["textual_inversion"]["args"]["mixed_precision"] = st.selectbox("Mixed Precision", ["no", "fp16", "bf16"], index=1,
																			   help="Whether to use mixed precision. Choose" "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
																			   "and an Nvidia Ampere GPU.")

							st.session_state["textual_inversion"]["args"]["local_rank"] = int(st.text_input("Local Rank",  value=1, help="For distributed training: local_rank"))

							st.session_state["textual_inversion"]["args"]["checkpoint_frequency"] = int(st.text_input("Checkpoint Frequency",  value=500, help="How often to save a checkpoint and sample image"))

							# stable_sample_batches is crashing when saving the samples so for now I will disable it util its fixed.
							#st.session_state["textual_inversion"]["args"]["stable_sample_batches"] = int(st.text_input("Stable Sample Batches",  value=0,
																						  #help="Number of fixed seed sample batches to generate per checkpoint"))

							st.session_state["textual_inversion"]["args"]["stable_sample_batches"] = 0

							st.session_state["textual_inversion"]["args"]["random_sample_batches"] = int(st.text_input("Random Sample Batches",  value=2,
																						  help="Number of random seed sample batches to generate per checkpoint"))

							st.session_state["textual_inversion"]["args"]["sample_batch_size"] = int(st.text_input("Sample Batch Size",  value=1, help="Number of samples to generate per batch"))

							st.session_state["textual_inversion"]["args"]["sample_steps"] = int(st.text_input("Sample Steps",  value=100,
																			 help="Number of steps for sample generation. Higher values will result in more detailed samples, but longer runtimes."))

							st.session_state["textual_inversion"]["args"]["custom_templates"] = st.text_input("Custom Templates",  value="",
																				 help="A semicolon-delimited list of custom template to use for samples, using {} as a placeholder for the concept.")
							with col5:
								st.session_state["textual_inversion"]["args"]["resume"] = st.checkbox(label="Resume Previous Run?", value=False,
								                                                                      help="Resume previous run, if a valid resume.json file is on the output dir \
								                                                                      it will be used, otherwise if the 'Resume From' field bellow contains a valid resume.json file \
								                                                                      that one will be used.")

								st.session_state["textual_inversion"]["args"]["resume_from"] = st.text_input(label="Resume From", help="Path to a directory to resume training from (ie, logs/token_name)")

								#st.session_state["textual_inversion"]["args"]["resume_checkpoint"] = st.file_uploader("Resume Checkpoint", type=["bin"],
																					  #help="Path to a specific checkpoint to resume training from (ie, logs/token_name/checkpoints/something.bin).")

								#st.session_state["textual_inversion"]["args"]["st.session_state["textual_inversion"]"] = st.file_uploader("st.session_state["textual_inversion"] File",  type=["json"],
																		#help="Path to a JSON st.session_state["textual_inversion"]uration file containing arguments for invoking this script."
																		#"If resume_from is given, its resume.json takes priority over this.")
			#
			#print (os.path.join(st.session_state["textual_inversion"]["args"]["output_dir"],st.session_state["textual_inversion"]["args"]["placeholder_token"].strip("<>"),"resume.json"))
			#print (os.path.exists(os.path.join(st.session_state["textual_inversion"]["args"]["output_dir"],st.session_state["textual_inversion"]["args"]["placeholder_token"].strip("<>"),"resume.json")))
			if os.path.exists(os.path.join(st.session_state["textual_inversion"]["args"]["output_dir"],st.session_state["textual_inversion"]["args"]["placeholder_token"].strip("<>"),"resume.json")):
				st.session_state["textual_inversion"]["args"]["resume_from"] = os.path.join(
				    st.session_state["textual_inversion"]["args"]["output_dir"], st.session_state["textual_inversion"]["args"]["placeholder_token"].strip("<>"))
				#print (st.session_state["textual_inversion"]["args"]["resume_from"])

			if os.path.exists(os.path.join(st.session_state["textual_inversion"]["args"]["output_dir"],st.session_state["textual_inversion"]["args"]["placeholder_token"].strip("<>"), "checkpoints","last.bin")):
				st.session_state["textual_inversion"]["args"]["resume_checkpoint"] = os.path.join(
			        st.session_state["textual_inversion"]["args"]["output_dir"], st.session_state["textual_inversion"]["args"]["placeholder_token"].strip("<>"), "checkpoints","last.bin")

			#if "resume_from" in st.session_state["textual_inversion"]["args"]:
				#if st.session_state["textual_inversion"]["args"]["resume_from"]:
					#if os.path.exists(os.path.join(st.session_state["textual_inversion"]['args']['resume_from'], "resume.json")):
						#with open(os.path.join(st.session_state["textual_inversion"]['args']['resume_from'], "resume.json"), 'rt') as f:
							#try:
								#resume_json = json.load(f)["args"]
								#st.session_state["textual_inversion"]["args"] = OmegaConf.merge(st.session_state["textual_inversion"]["args"], resume_json)
								#st.session_state["textual_inversion"]["args"]["resume_from"] = os.path.join(
								    #st.session_state["textual_inversion"]["args"]["output_dir"], st.session_state["textual_inversion"]["args"]["placeholder_token"].strip("<>"))
							#except json.decoder.JSONDecodeError:
								#pass

							#print(st.session_state["textual_inversion"]["args"])
							#print(st.session_state["textual_inversion"]["args"]['resume_from'])

			#elif st.session_state["textual_inversion"]["args"]["st.session_state["textual_inversion"]"] is not None:
				#with open(st.session_state["textual_inversion"]["args"]["st.session_state["textual_inversion"]"], 'rt') as f:
					#args = parser.parse_args(namespace=argparse.Namespace(**json.load(f)["args"]))

			env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
			if env_local_rank != -1 and env_local_rank != st.session_state["textual_inversion"]["args"]["local_rank"]:
				st.session_state["textual_inversion"]["args"]["local_rank"] = env_local_rank

			if st.session_state["textual_inversion"]["args"]["train_data_dir"] is None:
				st.error("You must specify --train_data_dir")

			if st.session_state["textual_inversion"]["args"]["pretrained_model_name_or_path"] is None:
				st.error("You must specify --pretrained_model_name_or_path")

			if st.session_state["textual_inversion"]["args"]["placeholder_token"] is None:
				st.error("You must specify --placeholder_token")

			if st.session_state["textual_inversion"]["args"]["initializer_token"] is None:
				st.error("You must specify --initializer_token")

			if st.session_state["textual_inversion"]["args"]["output_dir"] is None:
				st.error("You must specify --output_dir")

			# add a spacer and the submit button for the form.

			st.session_state["textual_inversion"]["message"] = st.empty()
			st.session_state["textual_inversion"]["progress_bar"] = st.empty()

			st.write("---")

			submit = st.form_submit_button("Run",help="")
			if submit:
				if "pipe" in st.session_state:
					del st.session_state["pipe"]
				if "model" in st.session_state:
					del st.session_state["model"]

				set_page_title("Running Textual Inversion - Stable Diffusion WebUI")
				#st.session_state["textual_inversion"]["message"].info("Textual Inversion Running. For more info check the progress on your console or the Ouput Tab.")

				try:
					#try:
					# run textual inversion.
					config = st.session_state['textual_inversion']
					textual_inversion(config)
					#except RuntimeError:
						#if "pipeline" in server_state["textual_inversion"]:
							#del server_state['textual_inversion']["checker"]
							#del server_state['textual_inversion']["unwrapped"]
							#del server_state['textual_inversion']["pipeline"]

						# run textual inversion.
						#config = st.session_state['textual_inversion']
						#textual_inversion(config)

					set_page_title("Textual Inversion - Stable Diffusion WebUI")

				except StopException:
					set_page_title("Textual Inversion - Stable Diffusion WebUI")
					print(f"Received Streamlit StopException")

				st.session_state["textual_inversion"]["message"].empty()

		#
		with output_tab:
			st.info("Under Construction. :construction_worker:")

			#st.info("Nothing to show yet. Maybe try running some training first.")

			#st.session_state["textual_inversion"]["preview_image"] = st.empty()
			#st.session_state["textual_inversion"]["progress_bar"] = st.empty()


		with tensorboard_tab:
			#st.info("Under Construction. :construction_worker:")

			# Start TensorBoard
			st_tensorboard(logdir=os.path.join("outputs", "textual_inversion"), port=8888)


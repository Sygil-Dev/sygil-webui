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
import argparse
import itertools
import math
import os
import random
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from pipelines.stable_diffusion.no_check import NoCheck
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from slugify import slugify
import json
import os
import sys

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=500,
        help="How often to save a checkpoint and sample image",
    )
    parser.add_argument(
        "--stable_sample_batches",
        type=int,
        default=0,
        help="Number of fixed seed sample batches to generate per checkpoint",
    )
    parser.add_argument(
        "--random_sample_batches",
        type=int,
        default=1,
        help="Number of random seed sample batches to generate per checkpoint",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Number of samples to generate per batch",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="Number of steps for sample generation. Higher values will result in more detailed samples, but longer runtimes.",
    )
    parser.add_argument(
        "--custom_templates",
        type=str,
        default=None,
        help=(
            "A semicolon-delimited list of custom template to use for samples, using {} as a placeholder for the concept."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a directory to resume training from (ie, logs/token_name/2022-09-22T23-36-27)"
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume training from (ie, logs/token_name/2022-09-22T23-36-27/checkpoints/something.bin)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON configuration file containing arguments for invoking this script. If resume_from is given, its resume.json takes priority over this."
    )

    args = parser.parse_args()
    if args.resume_from is not None:
        with open(f"{args.resume_from}/resume.json", 'rt') as f:
            args = parser.parse_args(namespace=argparse.Namespace(**json.load(f)["args"]))
    elif args.config is not None:
        with open(args.config, 'rt') as f:
            args = parser.parse_args(namespace=argparse.Namespace(**json.load(f)["args"]))

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify --train_data_dir")

    if args.pretrained_model_name_or_path is None:
        raise ValueError("You must specify --pretrained_model_name_or_path")

    if args.placeholder_token is None:
        raise ValueError("You must specify --placeholder_token")

    if args.initializer_token is None:
        raise ValueError("You must specify --initializer_token")

    if args.output_dir is None:
        raise ValueError("You must specify --output_dir")

    return args


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
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
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


def save_resume_file(basepath, args, extra = {}):
    info = {"args": vars(args)}
    info["args"].update(extra)
    with open(f"{basepath}/resume.json", "w") as f:
        json.dump(info, f, indent=4)

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
        samples_path = f"{self.output_dir}/samples"
        os.makedirs(samples_path, exist_ok=True)
        checker = NoCheck()

        unwrapped = self.accelerator.unwrap_model(text_encoder)
        # Save a sample image
        pipeline = StableDiffusionPipeline(
            text_encoder=unwrapped,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            ),
            safety_checker=NoCheck(),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        ).to("cuda")
        pipeline.enable_attention_slicing()

        if self.stable_sample_batches > 0:
            stable_latents = torch.randn(
                (self.sample_batch_size, pipeline.unet.in_channels, height // 8, width // 8),
                device=pipeline.device,
                generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
            )

            stable_prompts = [choice.format(self.placeholder_token) for choice in (self.templates * self.sample_batch_size)[:self.sample_batch_size]]

            # Generate and save stable samples
            for i in range(0, self.stable_sample_batches):
                samples = pipeline(
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
            samples = pipeline(
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

        del checker
        del unwrapped
        del pipeline
        torch.cuda.empty_cache()

def main():
    args = parse_args()

    global_step_offset = 0
    if args.resume_from is not None:
        basepath = f"{args.resume_from}"
        print("Resuming state from %s" % args.resume_from)
        with open(f"{basepath}/resume.json", 'r') as f:
            state = json.load(f)
        global_step_offset = state["args"].get("global_step", 0)

        print("We've trained %d steps so far" % global_step_offset)
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        basepath = f"{args.output_dir}/{slugify(args.placeholder_token)}/{now}"
        os.makedirs(basepath, exist_ok=True)


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path + '/tokenizer'
        )

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path + '/text_encoder',
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path + '/vae',
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path + '/unet',
    )

    base_templates = imagenet_style_templates_small if args.learnable_property == "style" else imagenet_templates_small
    if args.custom_templates:
        templates = args.custom_templates.split(";")
    else:
        templates = base_templates

    slice_size = unet.config.attention_head_dim // 2
    unet.set_attention_slice(slice_size)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data

    if args.resume_checkpoint is not None:
        token_embeds[placeholder_token_id] = torch.load(args.resume_checkpoint)[args.placeholder_token]
    else:
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    checkpointer = Checkpointer(
        accelerator=accelerator,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        placeholder_token=args.placeholder_token,
        placeholder_token_id=placeholder_token_id,
        templates=templates,
        output_dir=basepath,
        sample_batch_size=args.sample_batch_size,
        random_sample_batches=args.random_sample_batches,
        stable_sample_batches=args.stable_sample_batches,
        seed=args.seed
    )

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # TODO (patil-suraj): laod scheduler using args
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
    )

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
        templates=templates
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae and unet to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # Keep vae and unet in eval mode as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    encoded_pixel_values_cache = {}

    try:
        for epoch in range(args.num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    key = "|".join(batch["key"])
                    if encoded_pixel_values_cache.get(key, None) is None:
                        encoded_pixel_values_cache[key] = vae.encode(batch["pixel_values"]).latent_dist
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
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.checkpoint_frequency == 0 and global_step > 0 and accelerator.is_main_process:
                        checkpointer.checkpoint(global_step + global_step_offset, text_encoder)
                        save_resume_file(basepath, args, {
                            "global_step": global_step + global_step_offset,
                            "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
                        })
                        checkpointer.save_samples(
                            global_step + global_step_offset,
                            text_encoder,
                            args.resolution, args.resolution, 7.5, 0.0, args.sample_steps)

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                #accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            print("Finished! Saving final checkpoint and resume state.")
            checkpointer.checkpoint(
                global_step + global_step_offset,
                text_encoder,
                path=f"{basepath}/learned_embeds.bin"
            )

            save_resume_file(basepath, args, {
                "global_step": global_step + global_step_offset,
                "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
            })

            accelerator.end_training()

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("Interrupted, saving checkpoint and resume state...")
            checkpointer.checkpoint(global_step + global_step_offset, text_encoder)
            save_resume_file(basepath, args, {
                "global_step": global_step + global_step_offset,
                "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
            })
        quit()

if __name__ == "__main__":
    main()

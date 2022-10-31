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

"""
Implementation of Text to Video based on the
https://github.com/nateraw/stable-diffusion-videos
repo and the original gist script from
https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
"""
from sd_utils import *

# streamlit imports
from streamlit import StopException
from streamlit.elements import image as STImage

#streamlit components section
from streamlit_server_state import server_state, server_state_lock

#other imports

import os, sys, json
from PIL import Image
import torch
import numpy as np
import time, inspect, timeit
import torch
from torch import autocast
from io import BytesIO
import imageio
from slugify import slugify

from diffusers import StableDiffusionPipeline, DiffusionPipeline
#from stable_diffusion_videos import StableDiffusionWalkPipeline

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, \
     PNDMScheduler

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import deprecate
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from typing import Callable, List, Optional, Union
from pathlib import Path
from torchvision.transforms.functional import pil_to_tensor
import librosa
from PIL import Image
from torchvision.io import write_video


# streamlit components
from custom_components import sygil_suggestions

# Temp imports

# end of imports
#---------------------------------------------------------------------------------------------------------------

sygil_suggestions.init()

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

class plugin_info():
    plugname = "txt2vid"
    description = "Text to Image"
    isTab = True
    displayPriority = 1

#
# -----------------------------------------------------------------------------

def txt2vid_generation_callback(step: int, timestep: int, latents: torch.FloatTensor):
    #print ("test")
    #scale and decode the image latents with vae
    cond_latents_2 = 1 / 0.18215 * latents
    image = server_state["pipe"].vae.decode(cond_latents_2)

    # generate output numpy image as uint8
    image = torch.clamp((image["sample"] + 1.0) / 2.0, min=0.0, max=1.0)
    image2 = transforms.ToPILImage()(image.squeeze_(0))

    st.session_state["preview_image"].image(image2)

def get_timesteps_arr(audio_filepath, offset, duration, fps=30, margin=1.0, smooth=0.0):
    y, sr = librosa.load(audio_filepath, offset=offset, duration=duration)

    # librosa.stft hardcoded defaults...
    # n_fft defaults to 2048
    # hop length is win_length // 4
    # win_length defaults to n_fft
    D = librosa.stft(y, n_fft=2048, hop_length=2048 // 4, win_length=2048)

    # Extract percussive elements
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin)
    y_percussive = librosa.istft(D_percussive, length=len(y))

    # Get normalized melspectrogram
    spec_raw = librosa.feature.melspectrogram(y=y_percussive, sr=sr)
    spec_max = np.amax(spec_raw, axis=0)
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)

    # Resize cumsum of spec norm to our desired number of interpolation frames
    x_norm = np.linspace(0, spec_norm.shape[-1], spec_norm.shape[-1])
    y_norm = np.cumsum(spec_norm)
    y_norm /= y_norm[-1]
    x_resize = np.linspace(0, y_norm.shape[-1], int(duration*fps))

    T = np.interp(x_resize, x_norm, y_norm)

    # Apply smoothing
    return T * (1 - smooth) + np.linspace(0.0, 1.0, T.shape[0]) * smooth

#
def make_video_pyav(
    frames_or_frame_dir: Union[str, Path, torch.Tensor],
        audio_filepath: Union[str, Path] = None,
        fps: int = 30,
        audio_offset: int = 0,
        audio_duration: int = 2,
        sr: int = 22050,
        output_filepath: Union[str, Path] = "output.mp4",
        glob_pattern: str = "*.png",
        ):
    """
    TODO - docstring here

    frames_or_frame_dir: (Union[str, Path, torch.Tensor]):
        Either a directory of images, or a tensor of shape (T, C, H, W) in range [0, 255].
    """

    # Torchvision write_video doesn't support pathlib paths
    output_filepath = str(output_filepath)

    if isinstance(frames_or_frame_dir, (str, Path)):
        frames = None
        for img in sorted(Path(frames_or_frame_dir).glob(glob_pattern)):
            frame = pil_to_tensor(Image.open(img)).unsqueeze(0)
            frames = frame if frames is None else torch.cat([frames, frame])
    else:
        frames = frames_or_frame_dir

    # TCHW -> THWC
    frames = frames.permute(0, 2, 3, 1)

    if audio_filepath:
        # Read audio, convert to tensor
        audio, sr = librosa.load(audio_filepath, sr=sr, mono=True, offset=audio_offset, duration=audio_duration)
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        write_video(
                    output_filepath,
                        frames,
                        fps=fps,
                audio_array=audio_tensor,
            audio_fps=sr,
            audio_codec="aac",
            options={"crf": "10", "pix_fmt": "yuv420p"},
                )
    else:
        write_video(output_filepath, frames, fps=fps, options={"crf": "10", "pix_fmt": "yuv420p"})

    return output_filepath


class StableDiffusionWalkPipeline(DiffusionPipeline):
    r"""
    Pipeline for generating videos by interpolating  Stable Diffusion's latent space.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
            self,
                vae: AutoencoderKL,
                text_encoder: CLIPTextModel,
                tokenizer: CLIPTokenizer,
                unet: UNet2DConditionModel,
                scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
                safety_checker: StableDiffusionSafetyChecker,
                feature_extractor: CLIPFeatureExtractor,
                ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                            f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                                " file"
                        )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
                    vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unet,
                        scheduler=scheduler,
                        safety_checker=safety_checker,
                        feature_extractor=feature_extractor,
                )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
            self,
                prompt: Optional[Union[str, List[str]]] = None,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                negative_prompt: Optional[Union[str, List[str]]] = None,
                num_images_per_prompt: Optional[int] = 1,
                eta: float = 0.0,
                generator: Optional[torch.Generator] = None,
                latents: Optional[torch.FloatTensor] = None,
                output_type: Optional[str] = "pil",
                return_dict: bool = True,
                callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                callback_steps: Optional[int] = 1,
                text_embeddings: Optional[torch.FloatTensor] = None,
                **kwargs,
                ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*, defaults to `None`):
                The prompt or prompts to guide the image generation. If not provided, `text_embeddings` is required.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            text_embeddings (`torch.FloatTensor`, *optional*, defaults to `None`):
                Pre-generated text embeddings to be used as inputs for image generation. Can be used in place of
                `prompt` to avoid re-computing the embeddings. If not provided, the embeddings will be generated from
                the supplied `prompt`.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                    callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
                        ):
            raise ValueError(
                            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                                f" {type(callback_steps)}."
                        )

        if text_embeddings is None:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

            # get prompt text embeddings
            text_inputs = self.tokenizer(
                            prompt,
                                padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                return_tensors="pt",
                        )
            text_input_ids = text_inputs.input_ids

            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
                print(
                                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
            text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        else:
            batch_size = text_embeddings.shape[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                                        f" {type(prompt)}."
                                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                                        " the batch size of `prompt`."
                                )
            else:
                uncond_tokens = negative_prompt

            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                            uncond_tokens,
                                padding="max_length",
                                max_length=max_length,
                                truncation=True,
                                return_tensors="pt",
                        )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            if self.device.type == "mps":
                # randn does not exist on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                                    self.device
                                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            print ("test")

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                            self.device
                        )
            image, has_nsfw_concept = self.safety_checker(
                            images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
                        )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def generate_inputs(self, prompt_a, prompt_b, seed_a, seed_b, noise_shape, T, batch_size):
        embeds_a = self.embed_text(prompt_a)
        embeds_b = self.embed_text(prompt_b)

        latents_a = self.init_noise(seed_a, noise_shape)
        latents_b = self.init_noise(seed_b, noise_shape)

        batch_idx = 0
        embeds_batch, noise_batch = None, None
        for i, t in enumerate(T):
            embeds = torch.lerp(embeds_a, embeds_b, t)
            noise = slerp(device="cuda", t=float(t), v0=latents_a, v1=latents_b, DOT_THRESHOLD=0.9995)

            embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
            noise_batch = noise if noise_batch is None else torch.cat([noise_batch, noise])
            batch_is_ready = embeds_batch.shape[0] == batch_size or i + 1 == T.shape[0]
            if not batch_is_ready:
                continue
            yield batch_idx, embeds_batch, noise_batch
            batch_idx += 1
            del embeds_batch, noise_batch
            torch.cuda.empty_cache()
            embeds_batch, noise_batch = None, None

    def make_clip_frames(
            self,
                prompt_a: str,
                prompt_b: str,
                seed_a: int,
                seed_b: int,
                num_interpolation_steps: int = 5,
                save_path: Union[str, Path] = "outputs/",
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                eta: float = 0.0,
                height: int = 512,
                width: int = 512,
                upsample: bool = False,
                batch_size: int = 1,
                image_file_ext: str = ".png",
                T: np.ndarray = None,
                skip: int = 0,
                callback = None,
                callback_steps:int = 1,
                ):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        T = T if T is not None else np.linspace(0.0, 1.0, num_interpolation_steps)
        if T.shape[0] != num_interpolation_steps:
            raise ValueError(f"Unexpected T shape, got {T.shape}, expected dim 0 to be {num_interpolation_steps}")

        if upsample:
            if getattr(self, "upsampler", None) is None:
                self.upsampler = RealESRGANModel.from_pretrained("nateraw/real-esrgan")
            self.upsampler.to(self.device)

        batch_generator = self.generate_inputs(
                    prompt_a,
                        prompt_b,
                        seed_a,
                        seed_b,
                        (1, self.unet.in_channels, height // 8, width // 8),
                        T[skip:],
                        batch_size,
                )

        frame_index = skip
        for _, embeds_batch, noise_batch in batch_generator:
            with torch.autocast("cuda"):
                outputs = self(
                                    latents=noise_batch,
                                        text_embeddings=embeds_batch,
                                        height=height,
                                        width=width,
                                        guidance_scale=guidance_scale,
                                        eta=eta,
                                        num_inference_steps=num_inference_steps,
                                        output_type="pil" if not upsample else "numpy",
                                        callback=callback,
                                        callback_steps=callback_steps,
                                        )["images"]

                for image in outputs:
                    frame_filepath = save_path / (f"frame%06d{image_file_ext}" % frame_index)
                    image = image if not upsample else self.upsampler(image)
                    image.save(frame_filepath)
                    frame_index += 1

    def walk(
            self,
                prompts: Optional[List[str]] = None,
                seeds: Optional[List[int]] = None,
                num_interpolation_steps: Optional[Union[int, List[int]]] = 5,  # int or list of int
                output_dir: Optional[str] = "./dreams",
                name: Optional[str] = None,
                image_file_ext: Optional[str] = ".png",
                fps: Optional[int] = 30,
                num_inference_steps: Optional[int] = 50,
                guidance_scale: Optional[float] = 7.5,
                eta: Optional[float] = 0.0,
                height: Optional[int] = 512,
                width: Optional[int] = 512,
                upsample: Optional[bool] = False,
                batch_size: Optional[int] = 1,
                resume: Optional[bool] = False,
                audio_filepath: str = None,
                audio_start_sec: Optional[Union[int, float]] = None,
                margin: Optional[float] = 1.0,
                smooth: Optional[float] = 0.0,
                callback=None,
                callback_steps=1,
                ):
        """Generate a video from a sequence of prompts and seeds. Optionally, add audio to the
        video to interpolate to the intensity of the audio.

        Args:
            prompts (Optional[List[str]], optional):
                list of text prompts. Defaults to None.
            seeds (Optional[List[int]], optional):
                list of random seeds corresponding to prompts. Defaults to None.
            num_interpolation_steps (Union[int, List[int]], *optional*):
                How many interpolation steps between each prompt. Defaults to None.
            output_dir (Optional[str], optional):
                Where to save the video. Defaults to './dreams'.
            name (Optional[str], optional):
                Name of the subdirectory of output_dir. Defaults to None.
            image_file_ext (Optional[str], *optional*, defaults to '.png'):
                The extension to use when writing video frames.
            fps (Optional[int], *optional*, defaults to 30):
                The frames per second in the resulting output videos.
            num_inference_steps (Optional[int], *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (Optional[float], *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (Optional[float], *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            height (Optional[int], *optional*, defaults to 512):
                height of the images to generate.
            width (Optional[int], *optional*, defaults to 512):
                width of the images to generate.
            upsample (Optional[bool], *optional*, defaults to False):
                When True, upsamples images with realesrgan.
            batch_size (Optional[int], *optional*, defaults to 1):
                Number of images to generate at once.
            resume (Optional[bool], *optional*, defaults to False):
                When True, resumes from the last frame in the output directory based
                on available prompt config. Requires you to provide the `name` argument.
            audio_filepath (str, *optional*, defaults to None):
                Optional path to an audio file to influence the interpolation rate.
            audio_start_sec (Optional[Union[int, float]], *optional*, defaults to 0):
                Global start time of the provided audio_filepath.
            margin (Optional[float], *optional*, defaults to 1.0):
                Margin from librosa hpss to use for audio interpolation.
            smooth (Optional[float], *optional*, defaults to 0.0):
                Smoothness of the audio interpolation. 1.0 means linear interpolation.

        This function will create sub directories for each prompt and seed pair.

        For example, if you provide the following prompts and seeds:

        ```
        prompts = ['a dog', 'a cat', 'a bird']
        seeds = [1, 2, 3]
        num_interpolation_steps = 5
        output_dir = 'output_dir'
        name = 'name'
        fps = 5
        ```

        Then the following directories will be created:

        ```
        output_dir
        ├── name
        │   ├── name_000000
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000000.mp4
        │   ├── name_000001
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000001.mp4
        │   ├── ...
        │   ├── name.mp4
        |   |── prompt_config.json
        ```

        Returns:
            str: The resulting video filepath. This video includes all sub directories' video clips.
        """
        if (callback_steps is None) or (
                    callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
                        ):
            raise ValueError(
                            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                                f" {type(callback_steps)}."
                        )

        # init the output dir
        if type(prompts) == str:
            sanitized_prompt = slugify(prompts)
        else:
            sanitized_prompt = slugify(prompts[0])

        full_path = os.path.join(str(output_dir), str(sanitized_prompt))

        if len(full_path) > 220:
            sanitized_prompt = sanitized_prompt[:220-len(full_path)]
            full_path = os.path.join(output_dir, sanitized_prompt)

        os.makedirs(full_path, exist_ok=True)

        # Where the final video of all the clips combined will be saved
        output_filepath = os.path.join(full_path, f"{sanitized_prompt}.mp4")

        # If using same number of interpolation steps between, we turn into list
        if not resume and isinstance(num_interpolation_steps, int):
            num_interpolation_steps = [num_interpolation_steps] * (len(prompts) - 1)

        if not resume:
            audio_start_sec = audio_start_sec or 0

        # Save/reload prompt config
        prompt_config_path = Path(os.path.join(full_path, "prompt_config.json"))
        if not resume:
            prompt_config_path.write_text(
                            json.dumps(
                                    dict(
                                            prompts=prompts,
                                                seeds=seeds,
                                                num_interpolation_steps=num_interpolation_steps,
                                                fps=fps,
                                                num_inference_steps=num_inference_steps,
                                                guidance_scale=guidance_scale,
                                                eta=eta,
                                                upsample=upsample,
                                                height=height,
                                                width=width,
                                                audio_filepath=audio_filepath,
                                                audio_start_sec=audio_start_sec,
                                                ),

                                        indent=2,
                                        sort_keys=False,
                                )
                        )
        else:
            data = json.load(open(prompt_config_path))
            prompts = data["prompts"]
            seeds = data["seeds"]
            num_interpolation_steps = data["num_interpolation_steps"]
            fps = data["fps"]
            num_inference_steps = data["num_inference_steps"]
            guidance_scale = data["guidance_scale"]
            eta = data["eta"]
            upsample = data["upsample"]
            height = data["height"]
            width = data["width"]
            audio_filepath = data["audio_filepath"]
            audio_start_sec = data["audio_start_sec"]

        for i, (prompt_a, prompt_b, seed_a, seed_b, num_step) in enumerate(
                    zip(prompts, prompts[1:], seeds, seeds[1:], num_interpolation_steps)
                        ):
            # {name}_000000 / {name}_000001 / ...
            save_path = Path(f"{full_path}/{name}_{i:06d}")

            # Where the individual clips will be saved
            step_output_filepath = Path(f"{save_path}/{name}_{i:06d}.mp4")

            # Determine if we need to resume from a previous run
            skip = 0
            if resume:
                if step_output_filepath.exists():
                    print(f"Skipping {save_path} because frames already exist")
                    continue

                existing_frames = sorted(save_path.glob(f"*{image_file_ext}"))
                if existing_frames:
                    skip = int(existing_frames[-1].stem[-6:]) + 1
                    if skip + 1 >= num_step:
                        print(f"Skipping {save_path} because frames already exist")
                        continue
                    print(f"Resuming {save_path.name} from frame {skip}")

            audio_offset = audio_start_sec + sum(num_interpolation_steps[:i]) / fps
            audio_duration = num_step / fps

            self.make_clip_frames(
                            prompt_a,
                                prompt_b,
                                seed_a,
                                seed_b,
                                num_interpolation_steps=num_step,
                                save_path=save_path,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                eta=eta,
                                height=height,
                                width=width,
                                upsample=upsample,
                                batch_size=batch_size,
                                skip=skip,
                                T=get_timesteps_arr(
                                    audio_filepath,
                                        offset=audio_offset,
                                duration=audio_duration,
                                fps=fps,
                                margin=margin,
                                smooth=smooth,
                                callback=callback,
                                callback_steps=callback_steps,
                                )
                                if audio_filepath
                                else None,
                        )
            make_video_pyav(
                            save_path,
                                audio_filepath=audio_filepath,
                        fps=fps,
                        output_filepath=step_output_filepath,
                        glob_pattern=f"*{image_file_ext}",
                        audio_offset=audio_offset,
                        audio_duration=audio_duration,
                        sr=44100,
                        )

        return make_video_pyav(
                    full_path,
                        audio_filepath=audio_filepath,
                        fps=fps,
                        audio_offset=audio_start_sec,
                        audio_duration=sum(num_interpolation_steps) / fps,
                        output_filepath=output_filepath,
                        glob_pattern=f"**/*{image_file_ext}",
                        sr=44100,
                )

    def embed_text(self, text):
        """Helper to embed some text"""
        with torch.autocast("cuda"):
            text_input = self.tokenizer(
                            text,
                                padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                    truncation=True,
                return_tensors="pt",
                        )
            with torch.no_grad():
                embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed

    def init_noise(self, seed, noise_shape):
        """Helper to initialize noise"""
        # randn does not exist on mps, so we create noise on CPU here and move it to the device after initialization
        if self.device.type == "mps":
            noise = torch.randn(
                            noise_shape,
                                device='cpu',
                                generator=torch.Generator(device='cpu').manual_seed(seed),
                    ).to(self.device)
        else:
            noise = torch.randn(
                            noise_shape,
                                device=self.device,
                                generator=torch.Generator(device=self.device).manual_seed(seed),
                        )
        return noise

    @classmethod
    def from_pretrained(cls, *args, tiled=False, **kwargs):
        """Same as diffusers `from_pretrained` but with tiled option, which makes images tilable"""
        if tiled:

            def patch_conv(**patch):
                cls = nn.Conv2d
                init = cls.__init__

                def __init__(self, *args, **kwargs):
                    return init(self, *args, **kwargs, **patch)

                cls.__init__ = __init__

            patch_conv(padding_mode="circular")

        pipeline = super().from_pretrained(*args, **kwargs)
        pipeline.tiled = tiled
        return pipeline

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
        st.session_state["update_preview_frequency_list"].append(st.session_state["update_preview_frequency"])


    try:
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


            #update the preview image if it is enabled and the frequency matches the step_counter
            if st.session_state["update_preview"]:
                step_counter += 1

                if step_counter == st.session_state["update_preview_frequency"]:
                    if st.session_state.dynamic_preview_frequency:
                        st.session_state["current_chunk_speed"],
                        st.session_state["previous_chunk_speed_list"],
                        st.session_state["update_preview_frequency"],
                        st.session_state["avg_update_preview_frequency"] = optimize_update_preview_frequency(st.session_state["current_chunk_speed"],
                                                                                                                                     st.session_state["previous_chunk_speed_list"],
                                                                                                                                     st.session_state["update_preview_frequency"],
                                                                                                                                     st.session_state["update_preview_frequency_list"])

                    #scale and decode the image latents with vae
                    cond_latents_2 = 1 / 0.18215 * cond_latents
                    image = pipe.vae.decode(cond_latents_2)

                    # generate output numpy image as uint8
                    image = torch.clamp((image["sample"] + 1.0) / 2.0, min=0.0, max=1.0)
                    image2 = transforms.ToPILImage()(image.squeeze_(0))

                    st.session_state["preview_image"].image(image2)

                    step_counter = 0

            duration = timeit.default_timer() - start

            st.session_state["current_chunk_speed"] = duration

            if duration >= 1:
                speed = "s/it"
            else:
                speed = "it/s"
                duration = 1 / duration

            #
            total_frames = (st.session_state.sampling_steps + st.session_state.num_inference_steps) * st.session_state.max_duration_in_seconds
            total_steps = st.session_state.sampling_steps + st.session_state.num_inference_steps

            if i > st.session_state.sampling_steps:
                inference_counter += 1
                inference_percent = int(100 * float(inference_counter + 1 if inference_counter < num_inference_steps else num_inference_steps)/float(num_inference_steps))
                inference_progress = f"{inference_counter + 1 if inference_counter < num_inference_steps else num_inference_steps}/{num_inference_steps} {inference_percent}% "
            else:
                inference_progress = ""

            total_percent = int(100 * float(i+1 if i+1 < (num_inference_steps + st.session_state.sampling_steps)
                                                        else (num_inference_steps + st.session_state.sampling_steps))/float((num_inference_steps + st.session_state.sampling_steps)))

            percent = int(100 * float(i+1 if i+1 < num_inference_steps else st.session_state.sampling_steps)/float(st.session_state.sampling_steps))
            frames_percent = int(100 * float(st.session_state.current_frame if st.session_state.current_frame < total_frames else total_frames)/float(total_frames))

            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].text(
                                    f"Running step: {i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps} "
                                        f"{percent if percent < 100 else 100}% {inference_progress}{duration:.2f}{speed} | "
                                        f"Frame: {st.session_state.current_frame + 1 if st.session_state.current_frame < total_frames else total_frames}/{total_frames} "
                                    f"{frames_percent if frames_percent < 100 else 100}% {st.session_state.frame_duration:.2f}{st.session_state.frame_speed}"
                                )

            if "progress_bar" in st.session_state:
                st.session_state["progress_bar"].progress(total_percent if total_percent < 100 else 100)

    except KeyError:
        raise StopException

    #scale and decode the image latents with vae
    cond_latents_2 = 1 / 0.18215 * cond_latents
    image = pipe.vae.decode(cond_latents_2)

    # generate output numpy image as uint8
    image = torch.clamp((image["sample"] + 1.0) / 2.0, min=0.0, max=1.0)
    image2 = transforms.ToPILImage()(image.squeeze_(0))


    return image2

#
def load_diffusers_model(weights_path,torch_device):

    with server_state_lock["model"]:
        if "model" in server_state:
            del server_state["model"]

    if "textual_inversion" in st.session_state:
        del st.session_state['textual_inversion']

    try:
        with server_state_lock["pipe"]:
            if "pipe" not in server_state:
                if "weights_path" in st.session_state and st.session_state["weights_path"] != weights_path:
                    del st.session_state["weights_path"]

                st.session_state["weights_path"] = weights_path
                server_state['float16'] = st.session_state['defaults'].general.use_float16
                server_state['no_half'] = st.session_state['defaults'].general.no_half
                server_state['optimized'] = st.session_state['defaults'].general.optimized

                #if folder "models/diffusers/stable-diffusion-v1-4" exists, load the model from there
                if weights_path == "CompVis/stable-diffusion-v1-4":
                    model_path = os.path.join("models", "diffusers", "stable-diffusion-v1-4")

                if weights_path == "runwayml/stable-diffusion-v1-5":
                    model_path = os.path.join("models", "diffusers", "stable-diffusion-v1-5")

                if not os.path.exists(model_path + "/model_index.json"):
                    server_state["pipe"] = StableDiffusionWalkPipeline.from_pretrained(
                                            weights_path,
                                                use_local_file=True,
                                                use_auth_token=st.session_state["defaults"].general.huggingface_token,
                                                torch_dtype=torch.float16 if st.session_state['defaults'].general.use_float16 else None,
                                                revision="fp16" if not st.session_state['defaults'].general.no_half else None,
                                                safety_checker=None,  # Very important for videos...lots of false positives while interpolating
                                                #custom_pipeline="interpolate_stable_diffusion",
                                        )

                    StableDiffusionWalkPipeline.save_pretrained(server_state["pipe"], model_path)
                else:
                    server_state["pipe"] = StableDiffusionWalkPipeline.from_pretrained(
                                            model_path,
                                                use_local_file=True,
                                                torch_dtype=torch.float16 if st.session_state['defaults'].general.use_float16 else None,
                                                revision="fp16" if not st.session_state['defaults'].general.no_half else None,
                                                safety_checker=None,  # Very important for videos...lots of false positives while interpolating
                                                #custom_pipeline="interpolate_stable_diffusion",
                                        )

                server_state["pipe"].unet.to(torch_device)
                server_state["pipe"].vae.to(torch_device)
                server_state["pipe"].text_encoder.to(torch_device)

                #if st.session_state.defaults.general.enable_attention_slicing:
                server_state["pipe"].enable_attention_slicing()

                if st.session_state.defaults.general.enable_minimal_memory_usage:
                    server_state["pipe"].enable_minimal_memory_usage()

                logger.info("Tx2Vid Model Loaded")
            else:
                # if the float16 or no_half options have changed since the last time the model was loaded then we need to reload the model.
                if ("float16" in server_state and server_state['float16'] != st.session_state['defaults'].general.use_float16) \
                                   or ("no_half" in server_state and server_state['no_half'] != st.session_state['defaults'].general.no_half) \
                                   or ("optimized" in server_state and server_state['optimized'] != st.session_state['defaults'].general.optimized):

                    del server_state['float16']
                    del server_state['no_half']
                    with server_state_lock["pipe"]:
                        del server_state["pipe"]
                        torch_gc()

                    del server_state['optimized']

                    server_state['float16'] = st.session_state['defaults'].general.use_float16
                    server_state['no_half'] = st.session_state['defaults'].general.no_half
                    server_state['optimized'] = st.session_state['defaults'].general.optimized

                    load_diffusers_model(weights_path, torch_device)
                else:
                    logger.info("Tx2Vid Model already Loaded")

    except (EnvironmentError, OSError) as e:
        if "huggingface_token" not in st.session_state or st.session_state["defaults"].general.huggingface_token == "None":
            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].error(
                    "You need a huggingface token in order to use the Text to Video tab. Use the Settings page to add your token under the Huggingface section. "
                    "Make sure you save your settings after adding it."
                )
            raise OSError("You need a huggingface token in order to use the Text to Video tab. Use the Settings page to add your token under the Huggingface section. "
                          "Make sure you save your settings after adding it.")
        else:
            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].error(e)

#
def save_video_to_disk(frames, seeds, sanitized_prompt, fps=6,save_video=True, outdir='outputs'):
    if save_video:
        # write video to memory
        #output = io.BytesIO()
        #writer = imageio.get_writer(os.path.join(os.getcwd(), st.session_state['defaults'].general.outdir, "txt2vid"), im, extension=".mp4", fps=30)
        #try:
        video_path = os.path.join(os.getcwd(), outdir, "txt2vid",f"{seeds}_{sanitized_prompt}{datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())[:8]}.mp4")
        writer = imageio.get_writer(video_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)

        writer.close()
        #except:
        #	print("Can't save video, skipping.")

    return video_path
#
def txt2vid(
    # --------------------------------------
        # args you probably want to change
        prompts = ["blueberry spaghetti", "strawberry spaghetti"], # prompt to dream about
        gpu:int = st.session_state['defaults'].general.gpu, # id of the gpu to run on
        #name:str = 'test', # name of this project, for the output directory
        #rootdir:str = st.session_state['defaults'].general.outdir,
        num_steps:int = 200, # number of steps between each pair of sampled points
        max_duration_in_seconds:int = 30, # number of frames to write and then exit the script
        num_inference_steps:int = 50, # more (e.g. 100, 200 etc) can create slightly better images
        cfg_scale:float = 5.0, # can depend on the prompt. usually somewhere between 3-10 is good
        save_video = True,
        save_video_on_stop = False,
        outdir='outputs',
        do_loop = False,
        use_lerp_for_text = False,
        seeds = None,
        quality:int = 100, # for jpeg compression of the output images
        eta:float = 0.0,
        width:int = 256,
        height:int = 256,
        weights_path = "runwayml/stable-diffusion-v1-5",
        scheduler="klms",  # choices: default, ddim, klms
        disable_tqdm = False,
        #-----------------------------------------------
        beta_start = 0.0001,
        beta_end = 0.00012,
        beta_schedule = "scaled_linear",
        starting_image=None,
        #-----------------------------------------------
        # from new version
        image_file_ext: Optional[str] = ".png",
        fps: Optional[int] = 30,
        upsample: Optional[bool] = False,
        batch_size: Optional[int] = 1,
        resume: Optional[bool] = False,
        audio_filepath: str = None,
        audio_start_sec: Optional[Union[int, float]] = None,
        margin: Optional[float] = 1.0,
        smooth: Optional[float] = 0.0,
        ):
    """
    prompt = ["blueberry spaghetti", "strawberry spaghetti"], # prompt to dream about
    gpu:int = st.session_state['defaults'].general.gpu, # id of the gpu to run on
    #name:str = 'test', # name of this project, for the output directory
    #rootdir:str = st.session_state['defaults'].general.outdir,
    num_steps:int = 200, # number of steps between each pair of sampled points
    max_duration_in_seconds:int = 10000, # number of frames to write and then exit the script
    num_inference_steps:int = 50, # more (e.g. 100, 200 etc) can create slightly better images
    cfg_scale:float = 5.0, # can depend on the prompt. usually somewhere between 3-10 is good
    do_loop = False,
    use_lerp_for_text = False,
    seed = None,
    quality:int = 100, # for jpeg compression of the output images
    eta:float = 0.0,
    width:int = 256,
    height:int = 256,
    weights_path = "runwayml/stable-diffusion-v1-5",
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
    #max_duration_in_seconds +=1

    assert torch.cuda.is_available()
    assert height % 8 == 0 and width % 8 == 0
    torch.manual_seed(seeds)
    torch_device = f"cuda:{gpu}"

    if type(seeds) == list:
        prompts = [prompts] * len(seeds)
    else:
        seeds = [seeds, random.randint(0, 2**32 - 1)]

    if type(prompts) == list:
        # init the output dir
        sanitized_prompt = slugify(prompts[0])
    else:
        # init the output dir
        sanitized_prompt = slugify(prompts)

    full_path = os.path.join(os.getcwd(), st.session_state['defaults'].general.outdir, "txt2vid", "samples", sanitized_prompt)

    if len(full_path) > 220:
        sanitized_prompt = sanitized_prompt[:220-len(full_path)]
        full_path = os.path.join(os.getcwd(), st.session_state['defaults'].general.outdir, "txt2vid", "samples", sanitized_prompt)

    os.makedirs(full_path, exist_ok=True)

    # Write prompt info to file in output dir so we can keep track of what we did
    if st.session_state.write_info_files:
        with open(os.path.join(full_path , f'{slugify(str(seeds))}_config.json' if len(prompts) > 1 else "prompts_config.json"), "w") as outfile:
            outfile.write(json.dumps(
                            dict(
                                    prompts = prompts,
                                        gpu = gpu,
                                num_steps = num_steps,
                                max_duration_in_seconds = max_duration_in_seconds,
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

    with st.session_state["progress_bar_text"].container():
        with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders,index=[0]):
            load_diffusers_model(weights_path, torch_device)

    if "pipe" not in server_state:
        logger.error('wtf')

    server_state["pipe"].scheduler = SCHEDULERS[scheduler]

    server_state["pipe"].use_multiprocessing_for_evaluation = False
    server_state["pipe"].use_multiprocessed_decoding = False

    #if do_loop:
        ##Makes the last prompt loop back to first prompt
        #prompts = [prompts, prompts]
        #seeds = [seeds, seeds]
        #first_seed, *seeds = seeds
        #prompts.append(prompts)
        #seeds.append(first_seed)

    with torch.autocast('cuda'):
        # get the conditional text embeddings based on the prompt
        text_input = server_state["pipe"].tokenizer(prompts, padding="max_length", max_length=server_state["pipe"].tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings = server_state["pipe"].text_encoder(text_input.input_ids.to(torch_device) )[0]

    #
    if st.session_state.defaults.general.use_sd_concepts_library:

        prompt_tokens = re.findall('<([a-zA-Z0-9-]+)>', str(prompts))

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
    init1 = torch.randn((1, server_state["pipe"].unet.in_channels, height // 8, width // 8), device=torch_device)


    # iterate the loop
    frames = []
    frame_index = 0

    second_count = 1

    st.session_state["total_frames_avg_duration"] = []
    st.session_state["total_frames_avg_speed"] = []

    try:
        # code for the new StableDiffusionWalkPipeline implementation.
        start = timeit.default_timer()

        # preview image works but its not the right way to use this, this also do not work properly as it only makes one image and then exits.
        #with torch.autocast("cuda"):
            #StableDiffusionWalkPipeline.__call__(self=server_state["pipe"],
                                #prompt=prompts, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=cfg_scale,
                                #negative_prompt="", num_images_per_prompt=1, eta=0.0,
                                #callback=txt2vid_generation_callback, callback_steps=1,
                                #num_interpolation_steps=num_steps,
                                #fps=30,
                                #image_file_ext = ".png",
                                #output_dir=full_path,        # Where images/videos will be saved
                                ##name='animals_test',        # Subdirectory of output_dir where images/videos will be saved
                                #upsample = False,
                                ##do_loop=do_loop,           # Change to True if you want last prompt to loop back to first prompt
                                #resume = False,
                                #audio_filepath = None,
                                #audio_start_sec = None,
                                #margin = 1.0,
                                #smooth = 0.0,                                                             )

        # works correctly generating all frames but do not show the preview image
        # we also do not have control over the generation and cant stop it until the end of it.
        #with torch.autocast("cuda"):
            #video_path = server_state["pipe"].walk(
                            #prompts=prompts,
                                #seeds=seeds,
                                #num_interpolation_steps=num_steps,
                                #height=height,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
                                #width=width,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
                                #batch_size=4,
                                #fps=30,
                                #image_file_ext = ".png",
                                #eta = 0.0,
                                #output_dir=full_path,        # Where images/videos will be saved
                                ##name='test',        # Subdirectory of output_dir where images/videos will be saved
                                #guidance_scale=cfg_scale,         # Higher adheres to prompt more, lower lets model take the wheel
                                #num_inference_steps=num_inference_steps,     # Number of diffusion steps per image generated. 50 is good default
                                #upsample = False,
                                ##do_loop=do_loop,           # Change to True if you want last prompt to loop back to first prompt
                                #resume = False,
                                #audio_filepath = None,
                                #audio_start_sec = None,
                                #margin = 1.0,
                                #smooth = 0.0,
                                #callback=txt2vid_generation_callback, # our callback function will be called with the arguments callback(step, timestep, latents)
                                #callback_steps=1 # our callback function will be called once this many steps are processed in a single frame
                        #)

        # old code
        total_frames = (st.session_state.sampling_steps + st.session_state.num_inference_steps) * st.session_state.max_duration_in_seconds

        while second_count < max_duration_in_seconds:
            st.session_state["frame_duration"] = 0
            st.session_state["frame_speed"] = 0
            st.session_state["current_frame"] = frame_index

            #print(f"Second: {second_count+1}/{max_duration_in_seconds}")

            # sample the destination
            init2 = torch.randn((1, server_state["pipe"].unet.in_channels, height // 8, width // 8), device=torch_device)

            for i, t in enumerate(np.linspace(0, 1, num_steps)):
                start = timeit.default_timer()
                logger.info(f"COUNT: {frame_index+1}/{total_frames}")

                if use_lerp_for_text:
                    init = torch.lerp(init1, init2, float(t))
                else:
                    init = slerp(gpu, float(t), init1, init2)

                #init = slerp(gpu, float(t), init1, init2)

                with autocast("cuda"):
                    image = diffuse(server_state["pipe"], cond_embeddings, init, num_inference_steps, cfg_scale, eta)

                if st.session_state["save_individual_images"] and not st.session_state["use_GFPGAN"] and not st.session_state["use_RealESRGAN"]:
                    #im = Image.fromarray(image)
                    outpath = os.path.join(full_path, 'frame%06d.png' % frame_index)
                    image.save(outpath, quality=quality)

                    # send the image to the UI to update it
                    #st.session_state["preview_image"].image(im)

                    #append the frames to the frames list so we can use them later.
                    frames.append(np.asarray(image))


                #
                #try:
                #if st.session_state["use_GFPGAN"] and server_state["GFPGAN"] is not None and not st.session_state["use_RealESRGAN"]:
                if st.session_state["use_GFPGAN"] and server_state["GFPGAN"] is not None:
                    #print("Running GFPGAN on image ...")
                    if "progress_bar_text" in st.session_state:
                        st.session_state["progress_bar_text"].text("Running GFPGAN on image ...")
                    #skip_save = True # #287 >_>
                    torch_gc()
                    cropped_faces, restored_faces, restored_img = server_state["GFPGAN"].enhance(np.array(image)[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    gfpgan_sample = restored_img[:,:,::-1]
                    gfpgan_image = Image.fromarray(gfpgan_sample)

                    outpath = os.path.join(full_path, 'frame%06d.png' % frame_index)
                    gfpgan_image.save(outpath, quality=quality)

                    #append the frames to the frames list so we can use them later.
                    frames.append(np.asarray(gfpgan_image))
                    try:
                        st.session_state["preview_image"].image(gfpgan_image)
                    except KeyError:
                        logger.error ("Cant get session_state, skipping image preview.")
                #except (AttributeError, KeyError):
                    #print("Cant perform GFPGAN, skipping.")

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

        # save the video after the generation is done.
        video_path = save_video_to_disk(frames, seeds, sanitized_prompt, save_video=save_video, outdir=outdir)

    except StopException:
        if save_video_on_stop:
            logger.info("Streamlit Stop Exception Received. Saving video")
            video_path = save_video_to_disk(frames, seeds, sanitized_prompt, save_video=save_video, outdir=outdir)
        else:
            video_path = None


    #if video_path and "preview_video" in st.session_state:
        ## show video preview on the UI
        #st.session_state["preview_video"].video(open(video_path, 'rb').read())

    mem_max_used, mem_total = mem_mon.read_and_stop()
    time_diff = time.time()- start

    info = f"""
            {prompts}
            Sampling Steps: {num_steps}, Sampler: {scheduler}, CFG scale: {cfg_scale}, Seed: {seeds}, Max Duration In Seconds: {max_duration_in_seconds}""".strip()
    stats = f'''
            Took { round(time_diff, 2) }s total ({ round(time_diff/(max_duration_in_seconds),2) }s per image)
            Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    return video_path, seeds, info, stats

#
def layout():
    with st.form("txt2vid-inputs"):
        st.session_state["generation_mode"] = "txt2vid"

        input_col1, generate_col1 = st.columns([10,1])
        with input_col1:
            #prompt = st.text_area("Input Text","")
            placeholder = "A corgi wearing a top hat as an oil painting."
            prompt = st.text_area("Input Text","", placeholder=placeholder, height=54)
            sygil_suggestions.suggestion_area(placeholder)

        # Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
        generate_col1.write("")
        generate_col1.write("")
        generate_button = generate_col1.form_submit_button("Generate")

        # creating the page layout using columns
        col1, col2, col3 = st.columns([2,5,2], gap="large")

        with col1:
            width = st.slider("Width:", min_value=st.session_state['defaults'].txt2vid.width.min_value, max_value=st.session_state['defaults'].txt2vid.width.max_value,
                                          value=st.session_state['defaults'].txt2vid.width.value, step=st.session_state['defaults'].txt2vid.width.step)
            height = st.slider("Height:", min_value=st.session_state['defaults'].txt2vid.height.min_value, max_value=st.session_state['defaults'].txt2vid.height.max_value,
                                           value=st.session_state['defaults'].txt2vid.height.value, step=st.session_state['defaults'].txt2vid.height.step)
            cfg_scale = st.number_input("CFG (Classifier Free Guidance Scale):", min_value=st.session_state['defaults'].txt2vid.cfg_scale.min_value,
                                                    value=st.session_state['defaults'].txt2vid.cfg_scale.value,
                                                    step=st.session_state['defaults'].txt2vid.cfg_scale.step,
                                                    help="How strongly the image should follow the prompt.")

            #uploaded_images = st.file_uploader("Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "webp"],
                                                #help="Upload an image which will be used for the image to image generation.")
            seed = st.text_input("Seed:", value=st.session_state['defaults'].txt2vid.seed, help=" The seed to use, if left blank a random seed will be generated.")
            #batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=st.session_state['defaults'].txt2vid.batch_count,
            # step=1, help="How many iterations or batches of images to generate in total.")
            #batch_size = st.slider("Batch size", min_value=1, max_value=250, value=st.session_state['defaults'].txt2vid.batch_size, step=1,
                    #help="How many images are at once in a batch.\
                    #It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
                    #Default: 1")

            st.session_state["max_duration_in_seconds"] = st.number_input("Max Duration In Seconds:", value=st.session_state['defaults'].txt2vid.max_duration_in_seconds,
                                                                                      help="Specify the max duration in seconds you want your video to be.")

            with st.expander("Preview Settings"):
                #st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=st.session_state['defaults'].txt2vid.update_preview,
                                                                    #help="If enabled the image preview will be updated during the generation instead of at the end. \
                                            #You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
                                            #By default this is enabled and the frequency is set to 1 step.")

                st.session_state["update_preview"] = st.session_state["defaults"].general.update_preview
                st.session_state["update_preview_frequency"] = st.number_input("Update Image Preview Frequency",
                                                                                               min_value=0,
                                                                                               value=st.session_state['defaults'].txt2vid.update_preview_frequency,
                                                                                               help="Frequency in steps at which the the preview image is updated. By default the frequency \
				                                                               is set to 1 step.")

                st.session_state["dynamic_preview_frequency"] = st.checkbox("Dynamic Preview Frequency", value=st.session_state['defaults'].txt2vid.dynamic_preview_frequency,
                                                                                            help="This option tries to find the best value at which we can update \
					                                               the preview image during generation while minimizing the impact it has in performance. Default: True")


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
            preview_video = st.session_state["preview_video"]

            message = st.empty()

        with gallery_tab:
            st.write('Here should be the image gallery, if I could make a grid in streamlit.')

    with col3:
        # If we have custom models available on the "models/custom"
        #folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
        custom_models_available()
        if server_state["CustomModel_available"]:
            custom_model = st.selectbox("Custom Model:", st.session_state["defaults"].txt2vid.custom_models_list,
                                                    index=st.session_state["defaults"].txt2vid.custom_models_list.index(st.session_state["defaults"].txt2vid.default_model),
                                                    help="Select the model you want to use. This option is only available if you have custom models \
				                            on your 'models/custom' folder. The model name that will be shown here is the same as the name\
				                            the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
				                        will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.5")
        else:
            custom_model = "runwayml/stable-diffusion-v1-5"

        #st.session_state["weights_path"] = custom_model
        #else:
            #custom_model = "runwayml/stable-diffusion-v1-5"
            #st.session_state["weights_path"] = f"CompVis/{slugify(custom_model.lower())}"

        st.session_state.sampling_steps = st.number_input("Sampling Steps", value=st.session_state['defaults'].txt2vid.sampling_steps.value,
                                                                  min_value=st.session_state['defaults'].txt2vid.sampling_steps.min_value,
                                                                  step=st.session_state['defaults'].txt2vid.sampling_steps.step, help="Number of steps between each pair of sampled points")

        st.session_state.num_inference_steps = st.number_input("Inference Steps:", value=st.session_state['defaults'].txt2vid.num_inference_steps.value,
                                                                       min_value=st.session_state['defaults'].txt2vid.num_inference_steps.min_value,
                                                                       step=st.session_state['defaults'].txt2vid.num_inference_steps.step,
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
            with st.expander("Output Settings"):
                st.session_state["separate_prompts"] = st.checkbox("Create Prompt Matrix.", value=st.session_state['defaults'].txt2vid.separate_prompts,
                                                                                   help="Separate multiple prompts using the `|` character, and get all combinations of them.")
                st.session_state["normalize_prompt_weights"] = st.checkbox("Normalize Prompt Weights.",
                                                                                           value=st.session_state['defaults'].txt2vid.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")

                st.session_state["save_individual_images"] = st.checkbox("Save individual images.",
                                                                                         value=st.session_state['defaults'].txt2vid.save_individual_images,
                                                                                         help="Save each image generated before any filter or enhancement is applied.")

                st.session_state["save_video"] = st.checkbox("Save video",value=st.session_state['defaults'].txt2vid.save_video,
                                                                             help="Save a video with all the images generated as frames at the end of the generation.")

                save_video_on_stop = st.checkbox("Save video on Stop",value=st.session_state['defaults'].txt2vid.save_video_on_stop,
                                                                 help="Save a video with all the images generated as frames when we hit the stop button during a generation.")

                st.session_state["group_by_prompt"] = st.checkbox("Group results by prompt", value=st.session_state['defaults'].txt2vid.group_by_prompt,
                                                                                  help="Saves all the images with the same prompt into the same folder. When using a prompt \
				                                                  matrix each prompt combination will have its own folder.")

                st.session_state["write_info_files"] = st.checkbox("Write Info file", value=st.session_state['defaults'].txt2vid.write_info_files,
                                                                                   help="Save a file next to the image with informartion about the generation.")

                st.session_state["do_loop"] = st.checkbox("Do Loop", value=st.session_state['defaults'].txt2vid.do_loop,
                                                                          help="Loop the prompt making two prompts from a single one.")

                st.session_state["use_lerp_for_text"] = st.checkbox("Use Lerp Instead of Slerp", value=st.session_state['defaults'].txt2vid.use_lerp_for_text,
                                                                                    help="Uses torch.lerp() instead of slerp. When interpolating between related prompts. \
				                                                    e.g. 'a lion in a grassy meadow' -> 'a bear in a grassy meadow' tends to keep the meadow \
				                                                    the whole way through when lerped, but slerping will often find a path where the meadow \
				                                                    disappears in the middle")

                st.session_state["save_as_jpg"] = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].txt2vid.save_as_jpg, help="Saves the images as jpg instead of png.")

            #
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
                            st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].txt2vid.use_GFPGAN,
                                                                                                     help="Uses the GFPGAN model to improve faces after the generation.\
						                                                                 This greatly improve the quality and consistency of faces but uses\
						                                                                 extra VRAM. Disable if you need the extra VRAM.")

                            st.session_state["GFPGAN_model"] = st.selectbox("GFPGAN model", st.session_state["GFPGAN_models"],
                                                                                                        index=st.session_state["GFPGAN_models"].index(st.session_state['defaults'].general.GFPGAN_model))

                            #st.session_state["GFPGAN_strenght"] = st.slider("Effect Strenght", min_value=1, max_value=100, value=1, step=1, help='')

                        else:
                            st.session_state["use_GFPGAN"] = False

                    with upscaling_tab:
                        st.session_state['us_upscaling'] = st.checkbox("Use Upscaling", value=st.session_state['defaults'].txt2vid.use_upscaling)
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

                                    st.session_state["ldsr_sampling_steps"] = st.number_input("Sampling Steps", value=st.session_state['defaults'].txt2vid.LDSR_config.sampling_steps,
                                                                                                                                  help="")

                                    st.session_state["preDownScale"] = st.number_input("PreDownScale", value=st.session_state['defaults'].txt2vid.LDSR_config.preDownScale,
                                                                                                                           help="")

                                    st.session_state["postDownScale"] = st.number_input("postDownScale", value=st.session_state['defaults'].txt2vid.LDSR_config.postDownScale,
                                                                                                                            help="")

                                    downsample_method_list = ['Nearest', 'Lanczos']
                                    st.session_state["downsample_method"] = st.selectbox("Downsample Method", downsample_method_list,
                                                                                                                             index=downsample_method_list.index(st.session_state['defaults'].txt2vid.LDSR_config.downsample_method))

                            else:
                                st.session_state["use_LDSR"] = False
                                st.session_state["LDSR_model"] = "model"

            with st.expander("Variant"):
                st.session_state["variant_amount"] = st.number_input("Variant Amount:", value=st.session_state['defaults'].txt2vid.variant_amount.value,
                                                                                     min_value=st.session_state['defaults'].txt2vid.variant_amount.min_value,
                                                                                     max_value=st.session_state['defaults'].txt2vid.variant_amount.max_value,
                                                                                     step=st.session_state['defaults'].txt2vid.variant_amount.step)

                st.session_state["variant_seed"] = st.text_input("Variant Seed:", value=st.session_state['defaults'].txt2vid.seed,
                                                                                 help="The seed to use when generating a variant, if left blank a random seed will be generated.")

            #st.session_state["beta_start"] = st.slider("Beta Start:", value=st.session_state['defaults'].txt2vid.beta_start.value,
                                                        #min_value=st.session_state['defaults'].txt2vid.beta_start.min_value,
                                                        #max_value=st.session_state['defaults'].txt2vid.beta_start.max_value,
                                                        #step=st.session_state['defaults'].txt2vid.beta_start.step, format=st.session_state['defaults'].txt2vid.beta_start.format)
            #st.session_state["beta_end"] = st.slider("Beta End:", value=st.session_state['defaults'].txt2vid.beta_end.value,
                                                        #min_value=st.session_state['defaults'].txt2vid.beta_end.min_value, max_value=st.session_state['defaults'].txt2vid.beta_end.max_value,
                                                        #step=st.session_state['defaults'].txt2vid.beta_end.step, format=st.session_state['defaults'].txt2vid.beta_end.format)

    if generate_button:
        #print("Loading models")
        # load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.
        #load_models(False, st.session_state["use_GFPGAN"], True, st.session_state["RealESRGAN_model"])

        if st.session_state["use_GFPGAN"]:
            if "GFPGAN" in server_state:
                logger.info("GFPGAN already loaded")
            else:
                with col2:
                    with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders,index=[0]):
                        # Load GFPGAN
                        if os.path.exists(st.session_state["defaults"].general.GFPGAN_dir):
                            try:
                                load_GFPGAN()
                                logger.info("Loaded GFPGAN")
                            except Exception:
                                import traceback
                                logger.error("Error loading GFPGAN:", file=sys.stderr)
                                logger.error(traceback.format_exc(), file=sys.stderr)
        else:
            if "GFPGAN" in server_state:
                del server_state["GFPGAN"]

        #try:
        # run video generation
        video, seed, info, stats = txt2vid(prompts=prompt, gpu=st.session_state["defaults"].general.gpu,
                                                   num_steps=st.session_state.sampling_steps, max_duration_in_seconds=st.session_state.max_duration_in_seconds,
                                                   num_inference_steps=st.session_state.num_inference_steps,
                                                   cfg_scale=cfg_scale, save_video_on_stop=save_video_on_stop,
                                                   outdir=st.session_state["defaults"].general.outdir,
                                                   do_loop=st.session_state["do_loop"],
                                                   use_lerp_for_text=st.session_state["use_lerp_for_text"],
                                                   seeds=seed, quality=100, eta=0.0, width=width,
                                                   height=height, weights_path=custom_model, scheduler=scheduler_name,
                                                   disable_tqdm=False, beta_start=st.session_state['defaults'].txt2vid.beta_start.value,
                                                   beta_end=st.session_state['defaults'].txt2vid.beta_end.value,
                                                   beta_schedule=beta_scheduler_type, starting_image=None)

        if video and save_video_on_stop:
            # show video preview on the UI after we hit the stop button
            # currently not working as session_state is cleared on StopException
            preview_video.video(open(video, 'rb').read())

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

        #except (StopException, KeyError):
            #print(f"Received Streamlit StopException")



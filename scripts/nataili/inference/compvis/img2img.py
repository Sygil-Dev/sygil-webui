import os
import re
import sys
import k_diffusion as K
import tqdm
from contextlib import contextmanager, nullcontext
import skimage
import numpy as np
import PIL
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.kdiffusion import CFGMaskedDenoiser, KDiffusionSampler
from ldm.models.diffusion.plms import PLMSSampler
from nataili.util.cache import torch_gc
from nataili.util.check_prompt_length import check_prompt_length
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.image_grid import image_grid
from nataili.util.load_learned_embed_in_clip import load_learned_embed_in_clip
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int
from slugify import slugify
import PIL


class img2img:
    def __init__(self, model, device, output_dir, save_extension='jpg',
    output_file_path=False, load_concepts=False, concepts_dir=None,
    verify_input=True, auto_cast=True):
        self.model = model
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
        self.auto_cast = auto_cast
        self.device = device
        self.comments = []
        self.output_images = []
        self.info = ''
        self.stats = ''
        self.images = []

    def create_random_tensors(self, shape, seeds):
        xs = []
        for seed in seeds:
            torch.manual_seed(seed)

            # randn results depend on device; gpu and cpu get different results for same seed;
            # the way I see it, it's better to do this on CPU, so that everyone gets same result;
            # but the original script had it like this so i do not dare change it for now because
            # it will break everyone's seeds.
            xs.append(torch.randn(shape, device=self.device))
        x = torch.stack(xs)
        return x

    def process_prompt_tokens(self, prompt_tokens):
        # compviz codebase
        tokenizer = self.model.cond_stage_model.tokenizer
        text_encoder = self.model.cond_stage_model.transformer

        # diffusers codebase
        #tokenizer = pipe.tokenizer
        #text_encoder = pipe.text_encoder

        ext = ('.pt', '.bin')
        for token_name in prompt_tokens:
            embedding_path = os.path.join(self.concepts_dir, token_name)	
            if os.path.exists(embedding_path):
                for files in os.listdir(embedding_path):
                    if files.endswith(ext):
                        load_learned_embed_in_clip(f"{os.path.join(embedding_path, files)}", text_encoder, tokenizer, f"<{token_name}>")
            else:
                print(f"Concept {token_name} not found in {self.concepts_dir}")
                del tokenizer, text_encoder
                return
        del tokenizer, text_encoder

    def resize_image(self, resize_mode, im, width, height):
        LANCZOS = (PIL.Image.Resampling.LANCZOS if hasattr(PIL.Image, 'Resampling') else PIL.Image.LANCZOS)
        if resize_mode == "resize":
            res = im.resize((width, height), resample=LANCZOS)
        elif resize_mode == "crop":
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio > src_ratio else im.width * height // im.height
            src_h = height if ratio <= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        else:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio < src_ratio else im.width * height // im.height
            src_h = height if ratio >= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

        return res
    
        #
    # helper fft routines that keep ortho normalization and auto-shift before and after fft
    def _fft2(self, data):
        if data.ndim > 2: # has channels
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:,:,c]
                out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
                out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
        else: # one channel
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
            out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])

        return out_fft

    def _ifft2(self, data):
        if data.ndim > 2: # has channels
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:,:,c]
                out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
                out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
        else: # one channel
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
            out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])

        return out_ifft

    def _get_gaussian_window(self, width, height, std=3.14, mode=0):

        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        window = np.zeros((width, height))
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            if mode == 0:
                window[:, y] = np.exp(-(x**2+fy**2) * std)
            else:
                window[:, y] = (1/((x**2+1.) * (fy**2+1.))) ** (std/3.14) # hey wait a minute that's not gaussian

        return window

    def _get_masked_window_rgb(self, np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        for c in range(3):
            np_mask_rgb[:,:,c] = hardened[:]
        return np_mask_rgb

    def get_matched_noise(self, _np_src_image, np_mask_rgb, noise_q, color_variation):
        """
        Explanation:
        Getting good results in/out-painting with stable diffusion can be challenging.
        Although there are simpler effective solutions for in-painting, out-painting can be especially challenging because there is no color data
        in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.
        Provided here is my take on a potential solution to this problem.

        By taking a fourier transform of the masked src img we get a function that tells us the presence and orientation of each feature scale in the unmasked src.
        Shaping the init/seed noise for in/outpainting to the same distribution of feature scales, orientations, and positions increases output coherence
        by helping keep features aligned. This technique is applicable to any continuous generation task such as audio or video, each of which can
        be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased. For multi-channel data such as color
        or stereo sound the "color tone" or histogram of the seed noise can be matched to improve quality (using scikit-image currently)
        This method is quite robust and has the added benefit of being fast independently of the size of the out-painted area.
        The effects of this method include things like helping the generator integrate the pre-existing view distance and camera angle.

        Carefully managing color and brightness with histogram matching is also essential to achieving good coherence.

        noise_q controls the exponent in the fall-off of the distribution can be any positive number, lower values means higher detail (range > 0, default 1.)
        color_variation controls how much freedom is allowed for the colors/palette of the out-painted area (range 0..1, default 0.01)
        This code is provided as is under the Unlicense (https://unlicense.org/)
        Although you have no obligation to do so, if you found this code helpful please find it in your heart to credit me [parlance-zz].

        Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
        This code is part of a new branch of a discord bot I am working on integrating with diffusers (https://github.com/parlance-zz/g-diffuser-bot)

        """

        global DEBUG_MODE
        global TMP_ROOT_PATH

        width = _np_src_image.shape[0]
        height = _np_src_image.shape[1]
        num_channels = _np_src_image.shape[2]

        np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
        np_mask_grey = (np.sum(np_mask_rgb, axis=2)/3.)
        np_src_grey = (np.sum(np_src_image, axis=2)/3.)
        all_mask = np.ones((width, height), dtype=bool)
        img_mask = np_mask_grey > 1e-6
        ref_mask = np_mask_grey < 1e-3

        windowed_image = _np_src_image * (1.-self._get_masked_window_rgb(np_mask_grey))
        windowed_image /= np.max(windowed_image)
        windowed_image += np.average(_np_src_image) * np_mask_rgb# / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
        #windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) / (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
        #_save_debug_img(windowed_image, "windowed_src_img")

        src_fft = self._fft2(windowed_image) # get feature statistics from masked src img
        src_dist = np.absolute(src_fft)
        src_phase = src_fft / src_dist
        #_save_debug_img(src_dist, "windowed_src_dist")

        noise_window = self._get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
        noise_rgb = np.random.random_sample((width, height, num_channels))
        noise_grey = (np.sum(noise_rgb, axis=2)/3.)
        noise_rgb *= color_variation # the colorfulness of the starting noise is blended to greyscale with a parameter
        for c in range(num_channels):
            noise_rgb[:,:,c] += (1. - color_variation) * noise_grey

        noise_fft = self._fft2(noise_rgb)
        for c in range(num_channels):
            noise_fft[:,:,c] *= noise_window
        noise_rgb = np.real(self._ifft2(noise_fft))
        shaped_noise_fft = self._fft2(noise_rgb)
        shaped_noise_fft[:,:,:] = np.absolute(shaped_noise_fft[:,:,:])**2 * (src_dist ** noise_q) * src_phase # perform the actual shaping

        brightness_variation = 0.#color_variation # todo: temporarily tieing brightness variation to color variation for now
        contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

        # scikit-image is used for histogram matching, very convenient!
        shaped_noise = np.real(self._ifft2(shaped_noise_fft))
        shaped_noise -= np.min(shaped_noise)
        shaped_noise /= np.max(shaped_noise)
        shaped_noise[img_mask,:] = skimage.exposure.match_histograms(shaped_noise[img_mask,:]**1., contrast_adjusted_np_src[ref_mask,:], channel_axis=1)
        shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb
        #_save_debug_img(shaped_noise, "shaped_noise")

        matched_noise = np.zeros((width, height, num_channels))
        matched_noise = shaped_noise[:]
        #matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
        #matched_noise = _np_src_image[:] * (1. - np_mask_rgb) + matched_noise * np_mask_rgb

        #_save_debug_img(matched_noise, "matched_noise")

        """
        todo:
        color_variation doesnt have to be a single number, the overall color tone of the out-painted area could be param controlled
        """

        return np.clip(matched_noise, 0., 1.)

    def find_noise_for_image(self, model, device, init_image, prompt, steps=200, cond_scale=2.0, verbose=False, normalize=False, generation_callback=None):
        image = np.array(init_image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2. * image - 1.
        image = image.to(device)
        x = model.get_first_stage_encoding(model.encode_first_stage(image))

        uncond = model.get_learned_conditioning([''])
        cond = model.get_learned_conditioning([prompt])

        s_in = x.new_ones([x.shape[0]])
        dnw = K.external.CompVisDenoiser(model)
        sigmas = dnw.get_sigmas(steps).flip(0)

        if verbose:
            print(sigmas)

        for i in tqdm.trange(1, len(sigmas)):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
            cond_in = torch.cat([uncond, cond])

            c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

            if i == 1:
                t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
            else:
                t = dnw.sigma_to_t(sigma_in)

            eps = model.apply_model(x_in * c_in, t, cond=cond_in)
            denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

            denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

            if i == 1:
                d = (x - denoised) / (2 * sigmas[i])
            else:
                d = (x - denoised) / sigmas[i - 1]

            dt = sigmas[i] - sigmas[i - 1]
            x = x + d * dt

        return x / sigmas[-1]

    def generate(self, prompt: str, init_img=None, init_mask=None, mask_mode='mask', resize_mode='resize', noise_mode='seed',
      denoising_strength:float=0.8, ddim_steps=50, sampler_name='k_lms', n_iter=1, batch_size=1, cfg_scale=7.5, seed=None,
                height=512, width=512, save_individual_images: bool = True, save_grid: bool = True, ddim_eta:float = 0.0):
        seed = seed_to_int(seed)
        image_dict = {
            "seed": seed
        }
        # Init image is assumed to be a PIL image
        init_img = self.resize_image('resize', init_img, width, height)
        if sampler_name == 'PLMS':
            sampler = PLMSSampler(self.model)
        elif sampler_name == 'DDIM':
            sampler = DDIMSampler(self.model)
        elif sampler_name == 'k_dpm_2_a':
            sampler = KDiffusionSampler(self.model,'dpm_2_ancestral')
        elif sampler_name == 'k_dpm_2':
            sampler = KDiffusionSampler(self.model,'dpm_2')
        elif sampler_name == 'k_euler_a':
            sampler = KDiffusionSampler(self.model,'euler_ancestral')
        elif sampler_name == 'k_euler':
            sampler = KDiffusionSampler(self.model,'euler')
        elif sampler_name == 'k_heun':
            sampler = KDiffusionSampler(self.model,'heun')
        elif sampler_name == 'k_lms':
            sampler = KDiffusionSampler(self.model,'lms')
        else:
            raise Exception("Unknown sampler: " + sampler_name)

        torch_gc()
        def process_init_mask(init_mask: PIL.Image):
            if init_mask.mode == "RGBA":
                init_mask = init_mask.convert('RGBA')
                background = PIL.Image.new('RGBA', init_mask.size, (0, 0, 0))
                init_mask = PIL.Image.alpha_composite(background, init_mask)
                init_mask = init_mask.convert('RGB')
            return init_mask

        if mask_mode == "mask":
            if init_mask:
                init_mask = process_init_mask(init_mask)
        elif mask_mode == "invert":
            if init_mask:
                init_mask = process_init_mask(init_mask)
                init_mask = PIL.ImageOps.invert(init_mask)
        elif mask_mode == "alpha":
            init_img_transparency = init_img.split()[-1].convert('L')#.point(lambda x: 255 if x > 0 else 0, mode='1')
            init_mask = init_img_transparency
            init_mask = init_mask.convert("RGB")
            init_mask = self.resize_image(resize_mode, init_mask, width, height)
            init_mask = init_mask.convert("RGB")

        assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(denoising_strength * ddim_steps)

        if init_mask is not None and (noise_mode == "matched" or noise_mode == "find_and_matched") and init_img is not None:
            noise_q = 0.99
            color_variation = 0.0
            mask_blend_factor = 1.0

            np_init = (np.asarray(init_img.convert("RGB"))/255.0).astype(np.float64) # annoyingly complex mask fixing
            np_mask_rgb = 1. - (np.asarray(PIL.ImageOps.invert(init_mask).convert("RGB"))/255.0).astype(np.float64)
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

            noise_rgb = self.get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
            blend_mask_rgb = np.clip(np_mask_rgb_dilated,0.,1.) ** (mask_blend_factor)
            noised = noise_rgb[:]
            blend_mask_rgb **= (2.)
            noised = np_init[:] * (1. - blend_mask_rgb) + noised * blend_mask_rgb

            np_mask_grey = np.sum(np_mask_rgb, axis=2)/3.
            ref_mask = np_mask_grey < 1e-3

            all_mask = np.ones((height, width), dtype=bool)
            noised[all_mask,:] = skimage.exposure.match_histograms(noised[all_mask,:]**1., noised[ref_mask,:], channel_axis=1)

            init_img = PIL.Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")

        def init():
            image = init_img.convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)

            mask_channel = None
            if init_mask:
                alpha = self.resize_image(resize_mode, init_mask, width // 8, height // 8)
                mask_channel = alpha.split()[-1]

            mask = None
            if mask_channel is not None:
                mask = np.array(mask_channel).astype(np.float32) / 255.0
                mask = (1 - mask)
                mask = np.tile(mask, (4, 1, 1))
                mask = mask[None].transpose(0, 1, 2, 3)
                mask = torch.from_numpy(mask).to(self.model.device)

            init_image = 2. * image - 1.
            init_image = init_image.to(self.model.device)
            init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

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
                                                    'cond_scale': cfg_scale, 'mask': z_mask, 'x0': x0, 'xi': xi}, disable=False)
            else:

                x0, z_mask = init_data

                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
                z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc_steps]*batch_size).to(self.model.device))

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

        torch_gc()
        
        if self.load_concepts and self.concepts_dir is not None:
            prompt_tokens = re.findall('<([a-zA-Z0-9-]+)>', prompt)    
            if prompt_tokens:
                self.process_prompt_tokens(prompt_tokens)

        os.makedirs(self.output_dir, exist_ok=True)

        sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)

        if self.verify_input:
            try:
                check_prompt_length(self.model, prompt, self.comments)
            except:
                import traceback
                print("Error verifying input:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

            all_prompts = batch_size * n_iter * [prompt]
            all_seeds = [seed + x for x in range(len(all_prompts))]

        precision_scope = torch.autocast if self.auto_cast else nullcontext

        with torch.no_grad(), precision_scope("cuda"):
            for n in range(n_iter):
                print(f"Iteration: {n+1}/{n_iter}")
                prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
                seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

                uc = self.model.get_learned_conditioning(len(prompts) * [''])

                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                c = self.model.get_learned_conditioning(prompts)

                opt_C = 4
                opt_f = 8
                shape = [opt_C, height // opt_f, width // opt_f]

                x = self.create_random_tensors(shape, seeds=seeds)
                init_data = init()
                samples_ddim = sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                for i, x_sample in enumerate(x_samples_ddim):
                    sanitized_prompt = slugify(prompts[i])
                    full_path = os.path.join(os.getcwd(), sample_path)
                    sample_path_i = sample_path
                    base_count = get_next_sequence_number(sample_path_i)
                    filename = f"{base_count:05}-{ddim_steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[:200-len(full_path)]

                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    x_sample = x_sample.astype(np.uint8)
                    image = PIL.Image.fromarray(x_sample)
                    image_dict['image'] = image
                    self.images.append(image_dict)

                    if save_individual_images:
                        path = os.path.join(sample_path, filename + '.' + self.save_extension)
                        success = save_sample(image, filename, sample_path_i, self.save_extension)
                        if success:
                            if self.output_file_path:
                                self.output_images.append(path)
                            else:
                                self.output_images.append(image)
                        else:
                            return

        self.info = f"""
                {prompt}
                Steps: {ddim_steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}
                """.strip()
        self.stats = f'''
                '''

        for comment in self.comments:
            self.info += "\n\n" + comment

        torch_gc()

        del sampler

        return

import os
import re
import sys
from contextlib import contextmanager, nullcontext

import numpy as np
import PIL
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.kdiffusion import KDiffusionSampler
from ldm.models.diffusion.plms import PLMSSampler
from nataili.util.cache import torch_gc
from nataili.util.check_prompt_length import check_prompt_length
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.image_grid import image_grid
from nataili.util.load_learned_embed_in_clip import load_learned_embed_in_clip
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int
from slugify import slugify


class txt2img:
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

    def generate(self, prompt: str, ddim_steps=50, sampler_name='k_lms', n_iter=1, batch_size=1, cfg_scale=7.5, seed=None,
                height=512, width=512, save_individual_images: bool = True, save_grid: bool = True, ddim_eta:float = 0.0):
        seed = seed_to_int(seed)

        image_dict = {
            "seed": seed
        }
        negprompt = ''
        if '###' in prompt:
            prompt, negprompt = prompt.split('###', 1)
            prompt = prompt.strip()
            negprompt = negprompt.strip()

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

        def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
            samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=unconditional_conditioning, x_T=x)
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

                uc = self.model.get_learned_conditioning(len(prompts) * [negprompt])

                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                c = self.model.get_learned_conditioning(prompts)

                opt_C = 4
                opt_f = 8
                shape = [opt_C, height // opt_f, width // opt_f]

                x = self.create_random_tensors(shape, seeds=seeds)

                samples_ddim = sample(init_data=None, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

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

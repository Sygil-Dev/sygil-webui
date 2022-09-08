from util.imports import *
from util.torch_gc import *
from util.MemUsageMonitor import *
from util.oxlamon_matrix import *
from util.check_prompt_length import *
from util.seed_to_int import *
from util.misc_diffusion import *
from util.split_weighted_subprompts import *
from util.slerp import *
from util.get_next_sequence_number import *
from util.load_models import *
from util.save_sample import *
from util.image_grid import *
from util.draw_prompt_matrix import *
from textual_inversion.embeddings import *


def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, save_grid, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        fp=None, ddim_eta=0.0, normalize_prompt_weights=True, init_img=None, init_mask=None,
        keep_mask=False, mask_blur_strength=3, denoising_strength=0.75, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, jpg_sample=False,
        variant_amount=0.0, variant_seed=None, save_individual_images: bool = True):
	"""this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
	assert prompt is not None
	torch_gc()
	# start time after garbage collection (or before?)
	start_time = time.time()

	# We will use this date here later for the folder name, need to start_time if not need
	run_start_dt = datetime.datetime.now()

	mem_mon = MemUsageMonitor('MemMon')
	mem_mon.start()

	if hasattr(st.session_state["model"], "embedding_manager"):
		load_embeddings(fp)

	os.makedirs(outpath, exist_ok=True)

	sample_path = os.path.join(outpath, "samples")
	os.makedirs(sample_path, exist_ok=True)

	if not ("|" in prompt) and prompt.startswith("@"):
		prompt = prompt[1:]

	comments = []

	prompt_matrix_parts = []
	simple_templating = False
	add_original_image = not (use_RealESRGAN or use_GFPGAN)

	if prompt_matrix:
		if prompt.startswith("@"):
			simple_templating = True
			add_original_image = not (use_RealESRGAN or use_GFPGAN)
			all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(prompt, seed, n_iter, batch_size)
		else:
			all_prompts = []
			prompt_matrix_parts = prompt.split("|")
			combination_count = 2 ** (len(prompt_matrix_parts) - 1)
			for combination_num in range(combination_count):
				current = prompt_matrix_parts[0]

				for n, text in enumerate(prompt_matrix_parts[1:]):
					if combination_num & (2 ** n) > 0:
						current += ("" if text.strip().startswith(",") else ", ") + text

				all_prompts.append(current)

			n_iter = math.ceil(len(all_prompts) / batch_size)
			all_seeds = len(all_prompts) * [seed]

		print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
	else:

		if not defaults.general.no_verify_input:
			try:
				check_prompt_length(prompt, comments)
			except:
				import traceback
				print("Error verifying input:", file=sys.stderr)
				print(traceback.format_exc(), file=sys.stderr)

		all_prompts = batch_size * n_iter * [prompt]
		all_seeds = [seed + x for x in range(len(all_prompts))]

	precision_scope = autocast if defaults.general.precision == "autocast" else nullcontext
	output_images = []
	grid_captions = []
	stats = []
	with torch.no_grad(), precision_scope("cuda"), (st.session_state["model"].ema_scope() if not defaults.general.optimized else nullcontext()):
		init_data = func_init()
		tic = time.time()


		# if variant_amount > 0.0 create noise from base seed
		base_x = None
		if variant_amount > 0.0:
			target_seed_randomizer = seed_to_int('') # random seed
			torch.manual_seed(seed) # this has to be the single starting seed (not per-iteration)
			base_x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=[seed])
			# we don't want all_seeds to be sequential from starting seed with variants, 
			# since that makes the same variants each time, 
			# so we add target_seed_randomizer as a random offset 
			for si in range(len(all_seeds)):
				all_seeds[si] += target_seed_randomizer

		for n in range(n_iter):
			print(f"Iteration: {n+1}/{n_iter}")
			prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
			captions = prompt_matrix_parts[n * batch_size:(n + 1) * batch_size]
			seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

			print(prompt)

			if defaults.general.optimized:
				modelCS.to(defaults.general.gpu)

			uc = (st.session_state["model"] if not defaults.general.optimized else modelCS).get_learned_conditioning(len(prompts) * [""])

			if isinstance(prompts, tuple):
				prompts = list(prompts)

			# split the prompt if it has : for weighting
			# TODO for speed it might help to have this occur when all_prompts filled??
			weighted_subprompts = split_weighted_subprompts(prompts[0], normalize_prompt_weights)

			# sub-prompt weighting used if more than 1
			if len(weighted_subprompts) > 1:
				c = torch.zeros_like(uc) # i dont know if this is correct.. but it works
				for i in range(0, len(weighted_subprompts)):
					# note if alpha negative, it functions same as torch.sub
					c = torch.add(c, (st.session_state["model"] if not defaults.general.optimized else modelCS).get_learned_conditioning(weighted_subprompts[i][0]), alpha=weighted_subprompts[i][1])
			else: # just behave like usual
				c = (st.session_state["model"] if not defaults.general.optimized else modelCS).get_learned_conditioning(prompts)


			shape = [opt_C, height // opt_f, width // opt_f]

			if defaults.general.optimized:
				mem = torch.cuda.memory_allocated()/1e6
				modelCS.to("cpu")
				while(torch.cuda.memory_allocated()/1e6 >= mem):
					time.sleep(1)

			if variant_amount == 0.0:
				# we manually generate all input noises because each one should have a specific seed
				x = create_random_tensors(shape, seeds=seeds)

			else: # we are making variants
				# using variant_seed as sneaky toggle, 
				# when not None or '' use the variant_seed
				# otherwise use seeds
				if variant_seed != None and variant_seed != '':
					specified_variant_seed = seed_to_int(variant_seed)
					torch.manual_seed(specified_variant_seed)
					seeds = [specified_variant_seed]
				target_x = create_random_tensors(shape, seeds=seeds)
				# finally, slerp base_x noise to target_x noise for creating a variant
				x = slerp(defaults.general.gpu, max(0.0, min(1.0, variant_amount)), base_x, target_x)

			samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

			if defaults.general.optimized:
				modelFS.to(defaults.general.gpu)

			x_samples_ddim = (st.session_state["model"] if not defaults.general.optimized else modelFS).decode_first_stage(samples_ddim)
			x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

			for i, x_sample in enumerate(x_samples_ddim):				
				sanitized_prompt = slugify(prompts[i])

				if sort_samples:
					full_path = os.path.join(os.getcwd(), sample_path, sanitized_prompt)


					sanitized_prompt = sanitized_prompt[:220-len(full_path)]
					sample_path_i = os.path.join(sample_path, sanitized_prompt)

					#print(f"output folder length: {len(os.path.join(os.getcwd(), sample_path_i))}")
					#print(os.path.join(os.getcwd(), sample_path_i))

					os.makedirs(sample_path_i, exist_ok=True)
					base_count = get_next_sequence_number(sample_path_i)
					filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}"
				else:
					full_path = os.path.join(os.getcwd(), sample_path)
					sample_path_i = sample_path
					base_count = get_next_sequence_number(sample_path_i)
					filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[:220-len(full_path)] #same as before

				x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
				x_sample = x_sample.astype(np.uint8)
				image = Image.fromarray(x_sample)
				original_sample = x_sample
				original_filename = filename

				if use_GFPGAN and st.session_state["GFPGAN"] is not None and not use_RealESRGAN:
					#skip_save = True # #287 >_>
					torch_gc()
					cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
					gfpgan_sample = restored_img[:,:,::-1]
					gfpgan_image = Image.fromarray(gfpgan_sample)
					gfpgan_filename = original_filename + '-gfpgan'

					save_sample(gfpgan_image, sample_path_i, gfpgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback,
                                                    uses_random_seed_loopback, save_grid, sort_samples, sampler_name, ddim_eta,
                                                    n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images=False)

					output_images.append(gfpgan_image) #287
					if simple_templating:
						grid_captions.append( captions[i] + "\ngfpgan" )

				if use_RealESRGAN and st.session_state["RealESRGAN"] is not None and not use_GFPGAN:
					#skip_save = True # #287 >_>
					torch_gc()

					if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
						#try_loading_RealESRGAN(realesrgan_model_name)
						load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

					output, img_mode = st.session_state["RealESRGAN"].enhance(x_sample[:,:,::-1])
					esrgan_filename = original_filename + '-esrgan4x'
					esrgan_sample = output[:,:,::-1]
					esrgan_image = Image.fromarray(esrgan_sample)

					#save_sample(image, sample_path_i, original_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
							#normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
							#save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)

					save_sample(esrgan_image, sample_path_i, esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                                    save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images=False)

					output_images.append(esrgan_image) #287
					if simple_templating:
						grid_captions.append( captions[i] + "\nesrgan" )

				if use_RealESRGAN and st.session_state["RealESRGAN"] is not None and use_GFPGAN and st.session_state["GFPGAN"] is not None:
					#skip_save = True # #287 >_>
					torch_gc()
					cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
					gfpgan_sample = restored_img[:,:,::-1]

					if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
						#try_loading_RealESRGAN(realesrgan_model_name)
						load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

					output, img_mode = st.session_state["RealESRGAN"].enhance(gfpgan_sample[:,:,::-1])
					gfpgan_esrgan_filename = original_filename + '-gfpgan-esrgan4x'
					gfpgan_esrgan_sample = output[:,:,::-1]
					gfpgan_esrgan_image = Image.fromarray(gfpgan_esrgan_sample)						    

					save_sample(gfpgan_esrgan_image, sample_path_i, gfpgan_esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, False, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                                    save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images=False)

					output_images.append(gfpgan_esrgan_image) #287

					if simple_templating:
						grid_captions.append( captions[i] + "\ngfpgan_esrgan" )

				if save_individual_images:
					save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                                                    normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                                    save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images)

					if not use_GFPGAN or not use_RealESRGAN:
						output_images.append(image)

					#if add_original_image or not simple_templating:
						#output_images.append(image)
						#if simple_templating:
							#grid_captions.append( captions[i] )

				if defaults.general.optimized:
					mem = torch.cuda.memory_allocated()/1e6
					modelFS.to("cpu")
					while(torch.cuda.memory_allocated()/1e6 >= mem):
						time.sleep(1)

		if prompt_matrix or save_grid:
			if prompt_matrix:
				if simple_templating:
					grid = image_grid(output_images, n_iter, force_n_rows=frows, captions=grid_captions)
				else:
					grid = image_grid(output_images, n_iter, force_n_rows=1 << ((len(prompt_matrix_parts)-1)//2))
					try:
						grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
					except:
						import traceback
						print("Error creating prompt_matrix text:", file=sys.stderr)
						print(traceback.format_exc(), file=sys.stderr)
			else:
				grid = image_grid(output_images, batch_size)

			if grid and (batch_size > 1  or n_iter > 1):
				output_images.insert(0, grid)

			grid_count = get_next_sequence_number(outpath, 'grid-')
			grid_file = f"grid-{grid_count:05}-{seed}_{slugify(prompts[i].replace(' ', '_')[:220-len(full_path)])}.{grid_ext}"
			grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)

		toc = time.time()

	mem_max_used, mem_total = mem_mon.read_and_stop()
	time_diff = time.time()-start_time

	info = f"""
                {prompt}
                Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', Denoising strength: '+str(denoising_strength) if init_img is not None else ''}{', GFPGAN' if use_GFPGAN and st.session_state["GFPGAN"] is not None else ''}{', '+realesrgan_model_name if use_RealESRGAN and st.session_state["RealESRGAN"] is not None else ''}{', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()
	stats = f'''
                Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
                Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

	for comment in comments:
		info += "\n\n" + comment

	#mem_mon.stop()
	#del mem_mon
	torch_gc()

	return output_images, seed, info, stats


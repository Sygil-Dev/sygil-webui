from util.imports import *

def save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
                normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images):

	filename_i = os.path.join(sample_path_i, filename)

	if not jpg_sample:
		if defaults.general.save_metadata:
			metadata = PngInfo()
			metadata.add_text("SD:prompt", prompts[i])
			metadata.add_text("SD:seed", str(seeds[i]))
			metadata.add_text("SD:width", str(width))
			metadata.add_text("SD:height", str(height))
			metadata.add_text("SD:steps", str(steps))
			metadata.add_text("SD:cfg_scale", str(cfg_scale))
			metadata.add_text("SD:normalize_prompt_weights", str(normalize_prompt_weights))
			if init_img is not None:
				metadata.add_text("SD:denoising_strength", str(denoising_strength))
			metadata.add_text("SD:GFPGAN", str(use_GFPGAN and st.session_state["GFPGAN"] is not None))
			image.save(f"{filename_i}.png", pnginfo=metadata)
		else:
			image.save(f"{filename_i}.png")
	else:
		image.save(f"{filename_i}.jpg", 'jpeg', quality=100, optimize=True)

	if write_info_files:
		# toggles differ for txt2img vs. img2img:
		offset = 0 if init_img is None else 2
		toggles = []
		if prompt_matrix:
			toggles.append(0)
		if normalize_prompt_weights:
			toggles.append(1)
		if init_img is not None:
			if uses_loopback:
				toggles.append(2)
			if uses_random_seed_loopback:
				toggles.append(3)
		if save_individual_images:
			toggles.append(2 + offset)
		if save_grid:
			toggles.append(3 + offset)
		if sort_samples:
			toggles.append(4 + offset)
		if write_info_files:
			toggles.append(5 + offset)
		if use_GFPGAN:
			toggles.append(6 + offset)
		info_dict = dict(
                        target="txt2img" if init_img is None else "img2img",
                prompt=prompts[i], ddim_steps=steps, toggles=toggles, sampler_name=sampler_name,
                    ddim_eta=ddim_eta, n_iter=n_iter, batch_size=batch_size, cfg_scale=cfg_scale,
                        seed=seeds[i], width=width, height=height
                )
		if init_img is not None:
			# Not yet any use for these, but they bloat up the files:
			#info_dict["init_img"] = init_img
			#info_dict["init_mask"] = init_mask
			info_dict["denoising_strength"] = denoising_strength
			info_dict["resize_mode"] = resize_mode
		with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
			yaml.dump(info_dict, f, allow_unicode=True, width=10000)

	# render the image on the frontend
	st.session_state["preview_image"].image(image)    

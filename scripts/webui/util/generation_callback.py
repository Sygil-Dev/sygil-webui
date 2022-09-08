from util.imports import *

@retry(tries=5)
def generation_callback(img, i=0):

	try:
		if i == 0:	
			if img['i']: i = img['i']
	except TypeError:
		pass


	if i % int(defaults.general.update_preview_frequency) == 0 and defaults.general.update_preview:
		#print (img)
		#print (type(img))
		# The following lines will convert the tensor we got on img to an actual image we can render on the UI.
		# It can probably be done in a better way for someone who knows what they're doing. I don't.		
		#print (img,isinstance(img, torch.Tensor))
		if isinstance(img, torch.Tensor):
			x_samples_ddim = (st.session_state["model"] if not defaults.general.optimized else modelFS).decode_first_stage(img)          
		else:
			# When using the k Diffusion samplers they return a dict instead of a tensor that look like this:
			# {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised}			
			x_samples_ddim = (st.session_state["model"] if not defaults.general.optimized else modelFS).decode_first_stage(img["denoised"])

		x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)  

		pil_image = transforms.ToPILImage()(x_samples_ddim.squeeze_(0)) 			

		# update image on the UI so we can see the progress
		st.session_state["preview_image"].image(pil_image) 	

	# Show a progress bar so we can keep track of the progress even when the image progress is not been shown,
	# Dont worry, it doesnt affect the performance.	
	if st.session_state["generation_mode"] == "txt2img":
		percent = int(100 * float(i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps)/float(st.session_state.sampling_steps))
		st.session_state["progress_bar_text"].text(
                        f"Running step: {i+1 if i+1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps} {percent if percent < 100 else 100}%")
	else:
		round_sampling_steps = round(st.session_state.sampling_steps * st.session_state["denoising_strength"])
		percent = int(100 * float(i+1 if i+1 < round_sampling_steps else round_sampling_steps)/float(round_sampling_steps))
		st.session_state["progress_bar_text"].text(
                        f"""Running step: {i+1 if i+1 < round_sampling_steps else round_sampling_steps}/{round_sampling_steps} {percent if percent < 100 else 100}%""")

	st.session_state["progress_bar"].progress(percent if percent < 100 else 100)

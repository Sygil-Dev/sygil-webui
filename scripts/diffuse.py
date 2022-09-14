from webui_streamlit import st
from sd_utils import *
import torch
import numpy as np

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

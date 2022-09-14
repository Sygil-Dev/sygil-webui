from webui_streamlit import st
from sd_utils import *
from streamlit import StopException
from PIL import Image, ImageOps

from img2img import img2img

def layout():
	with st.form("img2img-inputs"):
		st.session_state["generation_mode"] = "img2img"
	
		img2img_input_col, img2img_generate_col = st.columns([10,1])
		with img2img_input_col:
			#prompt = st.text_area("Input Text","")
			prompt = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")
	
		# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
		img2img_generate_col.write("")
		img2img_generate_col.write("")
		generate_button = img2img_generate_col.form_submit_button("Generate")
	
	
		# creating the page layout using columns
		col1_img2img_layout, col2_img2img_layout, col3_img2img_layout = st.columns([1,2,2], gap="small")    
	
		with col1_img2img_layout:
			# If we have custom models available on the "models/custom" 
			#folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
			if st.session_state["CustomModel_available"]:
				st.session_state["custom_model"] = st.selectbox("Custom Model:", st.session_state["custom_models"],
									    index=st.session_state["custom_models"].index(defaults.general.default_model),
							    help="Select the model you want to use. This option is only available if you have custom models \
							    on your 'models/custom' folder. The model name that will be shown here is the same as the name\
							    the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
							    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4") 	
			else:
				st.session_state["custom_model"] = "Stable Diffusion v1.4"
				
				
			st.session_state["sampling_steps"] = st.slider("Sampling Steps", value=defaults.img2img.sampling_steps, min_value=1, max_value=500)
			
			sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
			st.session_state["sampler_name"] = st.selectbox("Sampling method",sampler_name_list, 
									index=sampler_name_list.index(defaults.img2img.sampler_name), help="Sampling method to use.")
	
			mask_mode_list = ["Mask", "Inverted mask", "Image alpha"]
			mask_mode = st.selectbox("Mask Mode", mask_mode_list,
								 help="Select how you want your image to be masked.\"Mask\" modifies the image where the mask is white.\n\
								 \"Inverted mask\" modifies the image where the mask is black. \"Image alpha\" modifies the image where the image is transparent."
								 )
			mask_mode = mask_mode_list.index(mask_mode)
	
			width = st.slider("Width:", min_value=64, max_value=1024, value=defaults.img2img.width, step=64)
			height = st.slider("Height:", min_value=64, max_value=1024, value=defaults.img2img.height, step=64)
			seed = st.text_input("Seed:", value=defaults.img2img.seed, help=" The seed to use, if left blank a random seed will be generated.")
			noise_mode_list = ["Seed", "Find Noise", "Matched Noise", "Find+Matched Noise"]
			noise_mode = st.selectbox(
						"Noise Mode", noise_mode_list,
						help=""
					)
			noise_mode = noise_mode_list.index(noise_mode)
			find_noise_steps = st.slider("Find Noise Steps", value=100, min_value=1, max_value=500)
			batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.img2img.batch_count, step=1,
								help="How many iterations or batches of images to generate in total.")
	
			#			
			with st.expander("Advanced"):
				separate_prompts = st.checkbox("Create Prompt Matrix.", value=defaults.img2img.separate_prompts,
									       help="Separate multiple prompts using the `|` character, and get all combinations of them.")
				normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=defaults.img2img.normalize_prompt_weights,
										       help="Ensure the sum of all weights add up to 1.0")
				loopback = st.checkbox("Loopback.", value=defaults.img2img.loopback, help="Use images from previous batch when creating next batch.")
				random_seed_loopback = st.checkbox("Random loopback seed.", value=defaults.img2img.random_seed_loopback, help="Random loopback seed")
				save_individual_images = st.checkbox("Save individual images.", value=defaults.img2img.save_individual_images,
										     help="Save each image generated before any filter or enhancement is applied.")
				save_grid = st.checkbox("Save grid",value=defaults.img2img.save_grid, help="Save a grid with all the images generated into a single image.")
				group_by_prompt = st.checkbox("Group results by prompt", value=defaults.img2img.group_by_prompt,
									      help="Saves all the images with the same prompt into the same folder. \
									      When using a prompt matrix each prompt combination will have its own folder.")
				write_info_files = st.checkbox("Write Info file", value=defaults.img2img.write_info_files, 
									       help="Save a file next to the image with informartion about the generation.")						
				save_as_jpg = st.checkbox("Save samples as jpg", value=defaults.img2img.save_as_jpg, help="Saves the images as jpg instead of png.")
	
				if st.session_state["GFPGAN_available"]:
					use_GFPGAN = st.checkbox("Use GFPGAN", value=defaults.img2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation.\
							This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
				else:
					use_GFPGAN = False
	
				if st.session_state["RealESRGAN_available"]:
					st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=defaults.img2img.use_RealESRGAN,
										     help="Uses the RealESRGAN model to upscale the images after the generation.\
							This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
					st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
				else:
					st.session_state["use_RealESRGAN"] = False
					st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"
	
				variant_amount = st.slider("Variant Amount:", value=defaults.img2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
				variant_seed = st.text_input("Variant Seed:", value=defaults.img2img.variant_seed,
									     help="The seed to use when generating a variant, if left blank a random seed will be generated.")
				cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.img2img.cfg_scale, step=0.5,
								      help="How strongly the image should follow the prompt.")
				batch_size = st.slider("Batch size", min_value=1, max_value=100, value=defaults.img2img.batch_size, step=1,
								       help="How many images are at once in a batch.\
									       It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish \
									       generation as more images are generated at once.\
									       Default: 1")
	
				st.session_state["denoising_strength"] = st.slider("Denoising Strength:", value=defaults.img2img.denoising_strength, 
												   min_value=0.01, max_value=1.0, step=0.01)
	
			with st.expander("Preview Settings"):
				st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=defaults.img2img.update_preview,
												 help="If enabled the image preview will be updated during the generation instead of at the end. \
												 You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
												 By default this is enabled and the frequency is set to 1 step.")
	
				st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=defaults.img2img.update_preview_frequency,
													     help="Frequency in steps at which the the preview image is updated. By default the frequency \
													  is set to 1 step.")						
	
		with col2_img2img_layout:
			editor_tab = st.tabs(["Editor"])
	
			editor_image = st.empty()
			st.session_state["editor_image"] = editor_image
	
			st.form_submit_button("Refresh")
	
			masked_image_holder = st.empty()
			image_holder = st.empty()
	
			uploaded_images = st.file_uploader(
						"Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg"],
						help="Upload an image which will be used for the image to image generation.",
					)
			if uploaded_images:
				image = Image.open(uploaded_images).convert('RGBA')
				new_img = image.resize((width, height))
				image_holder.image(new_img)
	
			mask_holder = st.empty()
	
			uploaded_masks = st.file_uploader(
						"Upload Mask", accept_multiple_files=False, type=["png", "jpg", "jpeg"],
						help="Upload an mask image which will be used for masking the image to image generation.",
					)
			if uploaded_masks:
				mask = Image.open(uploaded_masks)
				if mask.mode == "RGBA":
					mask = mask.convert('RGBA')
					background = Image.new('RGBA', mask.size, (0, 0, 0))
					mask = Image.alpha_composite(background, mask)
				mask = mask.resize((width, height))
				mask_holder.image(mask)
	
			if uploaded_images and uploaded_masks:
				if mask_mode != 2:
					final_img = new_img.copy()
					alpha_layer = mask.convert('L')
					strength = st.session_state["denoising_strength"]
					if mask_mode == 0:
						alpha_layer = ImageOps.invert(alpha_layer)
						alpha_layer = alpha_layer.point(lambda a: a * strength)
						alpha_layer = ImageOps.invert(alpha_layer)
					elif mask_mode == 1:
						alpha_layer = alpha_layer.point(lambda a: a * strength)
						alpha_layer = ImageOps.invert(alpha_layer)
	
					final_img.putalpha(alpha_layer)
	
					with masked_image_holder.container():
						st.text("Masked Image Preview")
						st.image(final_img)
	
	
			with col3_img2img_layout:
				result_tab = st.tabs(["Result"])
	
				# create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
				preview_image = st.empty()
				st.session_state["preview_image"] = preview_image
	
				#st.session_state["loading"] = st.empty()
	
				st.session_state["progress_bar_text"] = st.empty()
				st.session_state["progress_bar"] = st.empty()
	
	
				message = st.empty()
	
				#if uploaded_images:
					#image = Image.open(uploaded_images).convert('RGB')
					##img_array = np.array(image) # if you want to pass it to OpenCV
					#new_img = image.resize((width, height))
					#st.image(new_img, use_column_width=True)
	
	
		if generate_button:
			#print("Loading models")
			# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.
			load_models(False, use_GFPGAN, st.session_state["use_RealESRGAN"], st.session_state["RealESRGAN_model"], st.session_state["CustomModel_available"],
				    st.session_state["custom_model"])                
			
			if uploaded_images:
				image = Image.open(uploaded_images).convert('RGBA')
				new_img = image.resize((width, height))
				#img_array = np.array(image) # if you want to pass it to OpenCV
				new_mask = None
				if uploaded_masks:
					mask = Image.open(uploaded_masks).convert('RGBA')
					new_mask = mask.resize((width, height))
	
				try:
					output_images, seed, info, stats = img2img(prompt=prompt, init_info=new_img, init_info_mask=new_mask, mask_mode=mask_mode,
										   ddim_steps=st.session_state["sampling_steps"],
										   sampler_name=st.session_state["sampler_name"], n_iter=batch_count,
										   cfg_scale=cfg_scale, denoising_strength=st.session_state["denoising_strength"], variant_seed=variant_seed,
										   seed=seed, noise_mode=noise_mode, find_noise_steps=find_noise_steps, width=width, 
										   height=height, fp=defaults.general.fp, variant_amount=variant_amount, 
										   ddim_eta=0.0, write_info_files=write_info_files, RealESRGAN_model=st.session_state["RealESRGAN_model"],
										   separate_prompts=separate_prompts, normalize_prompt_weights=normalize_prompt_weights,
										   save_individual_images=save_individual_images, save_grid=save_grid, 
										   group_by_prompt=group_by_prompt, save_as_jpg=save_as_jpg, use_GFPGAN=use_GFPGAN,
										   use_RealESRGAN=st.session_state["use_RealESRGAN"] if not loopback else False, loopback=loopback
										   )
	
					#show a message when the generation is complete.
					message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")
	
				except (StopException, KeyError):
					print(f"Received Streamlit StopException")
	
				# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
				# use the current col2 first tab to show the preview_img and update it as its generated.
				#preview_image.image(output_images, width=750)

#on import run init

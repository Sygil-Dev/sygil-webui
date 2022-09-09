from util.imports import *
from util.css import *
from util.load_models import *
from txt2img.txt2img import *
from img2img.img2img import *

def layout():

	st.set_page_config(page_title="Stable Diffusion Playground", layout="wide", initial_sidebar_state="collapsed")

	with st.empty():
		# load css as an external file, function has an option to local or remote url. Potential use when running from cloud infra that might not have access to local path.
		load_css(True, 'frontend/css/streamlit.main.css')

	# check if the models exist on their respective folders
	if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
		GFPGAN_available = True
	else:
		GFPGAN_available = False

	if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{defaults.general.RealESRGAN_model}.pth")):
		RealESRGAN_available = True
	else:
		RealESRGAN_available = False	

	with st.sidebar:
		# we should use an expander and group things together when more options are added so the sidebar is not too messy.
		#with st.expander("Global Settings:"):
		st.write("Global Settings:")
		defaults.general.update_preview = st.checkbox("Update Image Preview", value=defaults.general.update_preview,
                                                              help="If enabled the image preview will be updated during the generation instead of at the end. You can use the Update Preview \
							      Frequency option bellow to customize how frequent it's updated. By default this is enabled and the frequency is set to 1 step.")
		defaults.general.update_preview_frequency = st.text_input("Update Image Preview Frequency", value=defaults.general.update_preview_frequency,
                                                                          help="Frequency in steps at which the the preview image is updated. By default the frequency is set to 1 step.")



	txt2img_tab, img2img_tab, txt2video, postprocessing_tab = st.tabs(["Text-to-Image Unified", "Image-to-Image Unified", "Text-to-Video","Post-Processing"])

	with txt2img_tab:		
		with st.form("txt2img-inputs"):
			st.session_state["generation_mode"] = "txt2img"

			input_col1, generate_col1 = st.columns([10,1])
			with input_col1:
				#prompt = st.text_area("Input Text","")
				prompt = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")

			# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
			generate_col1.write("")
			generate_col1.write("")
			generate_button = generate_col1.form_submit_button("Generate")

			# creating the page layout using columns
			col1, col2, col3 = st.columns([1,2,1], gap="large")    

			with col1:
				width = st.slider("Width:", min_value=64, max_value=1024, value=defaults.txt2img.width, step=64)
				height = st.slider("Height:", min_value=64, max_value=1024, value=defaults.txt2img.height, step=64)
				cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.txt2img.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")
				seed = st.text_input("Seed:", value=defaults.txt2img.seed, help=" The seed to use, if left blank a random seed will be generated.")
				batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.txt2img.batch_count, step=1, help="How many iterations or batches of images to generate in total.")
				#batch_size = st.slider("Batch size", min_value=1, max_value=250, value=defaults.txt2img.batch_size, step=1,
					#help="How many images are at once in a batch.\
					#It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
					#Default: 1")

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

					message = st.empty()

				with gallery_tab:
					st.write('Here should be the image gallery, if I could make a grid in streamlit.')

			with col3:
				st.session_state.sampling_steps = st.slider("Sampling Steps", value=defaults.txt2img.sampling_steps, min_value=1, max_value=250)
				
				sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
				sampler_name = st.selectbox("Sampling method", sampler_name_list,
                                            index=sampler_name_list.index(defaults.txt2img.default_sampler), help="Sampling method to use. Default: k_euler")  



				#basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

				#with basic_tab:
					#summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
						#help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")

				with st.expander("Advanced"):
					separate_prompts = st.checkbox("Create Prompt Matrix.", value=False, help="Separate multiple prompts using the `|` character, and get all combinations of them.")
					normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=defaults.txt2img.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
					save_individual_images = st.checkbox("Save individual images.", value=defaults.txt2img.save_individual_images, help="Save each image generated before any filter or enhancement is applied.")
					save_grid = st.checkbox("Save grid",value=defaults.txt2img.save_grid, help="Save a grid with all the images generated into a single image.")
					group_by_prompt = st.checkbox("Group results by prompt", value=defaults.txt2img.group_by_prompt,
                                                                      help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
					write_info_files = st.checkbox("Write Info file", value=defaults.txt2img.write_info_files, help="Save a file next to the image with informartion about the generation.")
					save_as_jpg = st.checkbox("Save samples as jpg", value=defaults.txt2img.save_as_jpg, help="Saves the images as jpg instead of png.")

					if GFPGAN_available:
						use_GFPGAN = st.checkbox("Use GFPGAN", value=defaults.txt2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation. This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
					else:
						use_GFPGAN = False

					if RealESRGAN_available:
						use_RealESRGAN = st.checkbox("Use RealESRGAN", value=defaults.txt2img.use_RealESRGAN, help="Uses the RealESRGAN model to upscale the images after the generation. This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
						RealESRGAN_model = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
					else:
						use_RealESRGAN = False
						RealESRGAN_model = "RealESRGAN_x4plus"

					variant_amount = st.slider("Variant Amount:", value=defaults.txt2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
					variant_seed = st.text_input("Variant Seed:", value=defaults.txt2img.seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")


			if generate_button:
				#print("Loading models")
				# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.		
				load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model)                

				try:
					output_images, seed, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, RealESRGAN_model, batch_count, 1, 
										   cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
										   save_grid, group_by_prompt, save_as_jpg, use_GFPGAN, use_RealESRGAN, RealESRGAN_model, fp=defaults.general.fp,
										   variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files)
	
					message.success('Done!', icon="✅")

				except (StopException, KeyError):
					print(f"Received Streamlit StopException")

				# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
				# use the current col2 first tab to show the preview_img and update it as its generated.
				#preview_image.image(output_images)

	with img2img_tab:		
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
				st.session_state["sampling_steps"] = st.slider("Sampling Steps", value=defaults.img2img.sampling_steps, min_value=1, max_value=250)
				st.session_state["sampler_name"] = st.selectbox("Sampling method", ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"],
                                                                                index=0, help="Sampling method to use. Default: k_lms")  				

				uploaded_images = st.file_uploader("Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg"],
                                                                   help="Upload an image which will be used for the image to image generation."
                                                                   )

				width = st.slider("Width:", min_value=64, max_value=1024, value=defaults.img2img.width, step=64)
				height = st.slider("Height:", min_value=64, max_value=1024, value=defaults.img2img.height, step=64)
				seed = st.text_input("Seed:", value=defaults.img2img.seed, help=" The seed to use, if left blank a random seed will be generated.")
				batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.img2img.batch_count, step=1, help="How many iterations or batches of images to generate in total.")

				#			
				with st.expander("Advanced"):
					separate_prompts = st.checkbox("Create Prompt Matrix.", value=defaults.img2img.separate_prompts, help="Separate multiple prompts using the `|` character, and get all combinations of them.")
					normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=defaults.img2img.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
					loopback = st.checkbox("Loopback.", value=defaults.img2img.loopback, help="Use images from previous batch when creating next batch.")
					random_seed_loopback = st.checkbox("Random loopback seed.", value=defaults.img2img.random_seed_loopback, help="Random loopback seed")
					save_individual_images = st.checkbox("Save individual images.", value=defaults.img2img.save_individual_images, help="Save each image generated before any filter or enhancement is applied.")
					save_grid = st.checkbox("Save grid",value=defaults.img2img.save_grid, help="Save a grid with all the images generated into a single image.")
					group_by_prompt = st.checkbox("Group results by prompt", value=defaults.img2img.group_by_prompt,
                                                                      help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
					write_info_files = st.checkbox("Write Info file", value=defaults.img2img.write_info_files, help="Save a file next to the image with informartion about the generation.")
					save_as_jpg = st.checkbox("Save samples as jpg", value=defaults.img2img.save_as_jpg, help="Saves the images as jpg instead of png.")

					if GFPGAN_available:
						use_GFPGAN = st.checkbox("Use GFPGAN", value=defaults.img2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation.\
						This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
					else:
						use_GFPGAN = False

					if RealESRGAN_available:
						use_RealESRGAN = st.checkbox("Use RealESRGAN", value=defaults.img2img.use_RealESRGAN, help="Uses the RealESRGAN model to upscale the images after the generation.\
						This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
						RealESRGAN_model = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
					else:
						use_RealESRGAN = False
						RealESRGAN_model = "RealESRGAN_x4plus"

					variant_amount = st.slider("Variant Amount:", value=defaults.img2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
					variant_seed = st.text_input("Variant Seed:", value=defaults.img2img.variant_seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")
					cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.img2img.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")
					batch_size = st.slider("Batch size", min_value=1, max_value=100, value=defaults.img2img.batch_size, step=1,
                                                               help="How many images are at once in a batch.\
								       It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
								       Default: 1")

					st.session_state["denoising_strength"] = st.slider("Denoising Strength:", value=defaults.img2img.denoising_strength, min_value=0.01, max_value=1.0, step=0.01)


			with col2_img2img_layout:
				editor_tab = st.tabs(["Editor"])

				editor_image = st.empty()
				st.session_state["editor_image"] = editor_image

				if uploaded_images:
					image = Image.open(uploaded_images).convert('RGB')
					#img_array = np.array(image) # if you want to pass it to OpenCV
					new_img = image.resize((width, height))
					st.image(new_img)


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
				load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model)                
				if uploaded_images:
					image = Image.open(uploaded_images).convert('RGB')
					new_img = image.resize((width, height))
					#img_array = np.array(image) # if you want to pass it to OpenCV

					try:
						output_images, seed, info, stats = img2img(prompt=prompt, init_info=new_img, ddim_steps=st.session_state["sampling_steps"],
											   sampler_name=st.session_state["sampler_name"], n_iter=batch_count,
											   cfg_scale=cfg_scale, denoising_strength=st.session_state["denoising_strength"], variant_seed=variant_seed,
											   seed=seed, width=width, height=height, fp=defaults.general.fp, variant_amount=variant_amount, 
											   ddim_eta=0.0, write_info_files=write_info_files, RealESRGAN_model=RealESRGAN_model,
											   separate_prompts=separate_prompts, normalize_prompt_weights=normalize_prompt_weights,
											   save_individual_images=save_individual_images, save_grid=save_grid, 
											   group_by_prompt=group_by_prompt, save_as_jpg=save_as_jpg, use_GFPGAN=use_GFPGAN,
											   use_RealESRGAN=use_RealESRGAN if not loopback else False, loopback=loopback
											   )
	
						#show a message when the generation is complete.
						message.success('Done!', icon="✅")

					except (StopException, KeyError):
						print(f"Received Streamlit StopException")

				# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
				# use the current col2 first tab to show the preview_img and update it as its generated.
				#preview_image.image(output_images, width=750)
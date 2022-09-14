from webui_streamlit import st
from sd_utils import *

from io import BytesIO

from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
from streamlit.elements import image as STImage

from txt2vid import txt2vid
if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
	GFPGAN_available = True
else:
	GFPGAN_available = False

if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{defaults.general.RealESRGAN_model}.pth")):
	RealESRGAN_available = True
else:
	RealESRGAN_available = False	

def layout():
	with st.form("txt2vid-inputs"):
		st.session_state["generation_mode"] = "txt2vid"
	
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
			width = st.slider("Width:", min_value=64, max_value=2048, value=defaults.txt2vid.width, step=64)
			height = st.slider("Height:", min_value=64, max_value=2048, value=defaults.txt2vid.height, step=64)
			cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=defaults.txt2vid.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")
			seed = st.text_input("Seed:", value=defaults.txt2vid.seed, help=" The seed to use, if left blank a random seed will be generated.")
			#batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=defaults.txt2vid.batch_count, step=1, help="How many iterations or batches of images to generate in total.")
			#batch_size = st.slider("Batch size", min_value=1, max_value=250, value=defaults.txt2vid.batch_size, step=1,
				#help="How many images are at once in a batch.\
				#It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
				#Default: 1")
				
			st.session_state["max_frames"] = int(st.text_input("Max Frames:", value=defaults.txt2vid.max_frames, help="Specify the max number of frames you want to generate."))
			
			with st.expander("Preview Settings"):
				st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=defaults.txt2vid.update_preview,
					                                         help="If enabled the image preview will be updated during the generation instead of at the end. \
					                                         You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
					                                         By default this is enabled and the frequency is set to 1 step.")
				
				st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=defaults.txt2vid.update_preview_frequency,
					                                                  help="Frequency in steps at which the the preview image is updated. By default the frequency \
					                                                  is set to 1 step.")						
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
	
				message = st.empty()
	
			with gallery_tab:
				st.write('Here should be the image gallery, if I could make a grid in streamlit.')
	
		with col3:
			# If we have custom models available on the "models/custom" 
			#folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
			#if CustomModel_available:
			custom_model = st.selectbox("Custom Model:", defaults.txt2vid.custom_models_list,
				                    index=defaults.txt2vid.custom_models_list.index(defaults.txt2vid.default_model),
				                    help="Select the model you want to use. This option is only available if you have custom models \
				                    on your 'models/custom' folder. The model name that will be shown here is the same as the name\
				                    the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
				                will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4") 	
				
			#st.session_state["weights_path"] = custom_model
			#else:
				#custom_model = "CompVis/stable-diffusion-v1-4"
				#st.session_state["weights_path"] = f"CompVis/{slugify(custom_model.lower())}"
				
			st.session_state.sampling_steps = st.slider("Sampling Steps", value=defaults.txt2vid.sampling_steps, min_value=10, step=10, max_value=500,
				                                    help="Number of steps between each pair of sampled points")
			st.session_state.num_inference_steps = st.slider("Inference Steps:", value=defaults.txt2vid.num_inference_steps, min_value=10,step=10, max_value=500,
				                                         help="Higher values (e.g. 100, 200 etc) can create better images.")
	
			#sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
			#sampler_name = st.selectbox("Sampling method", sampler_name_list,
						    #index=sampler_name_list.index(defaults.txt2vid.default_sampler), help="Sampling method to use. Default: k_euler")  
			scheduler_name_list = ["klms", "ddim"]
			scheduler_name = st.selectbox("Scheduler:", scheduler_name_list,
				                    index=scheduler_name_list.index(defaults.txt2vid.scheduler_name), help="Scheduler to use. Default: klms")  
			
			beta_scheduler_type_list = ["scaled_linear", "linear"]
			beta_scheduler_type = st.selectbox("Beta Schedule Type:", beta_scheduler_type_list,
				                    index=beta_scheduler_type_list.index(defaults.txt2vid.beta_scheduler_type), help="Schedule Type to use. Default: linear")  			
	
	
			#basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])
	
			#with basic_tab:
				#summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
					#help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")
	
			with st.expander("Advanced"):
				st.session_state["separate_prompts"] = st.checkbox("Create Prompt Matrix.", value=defaults.txt2vid.separate_prompts,
					                                           help="Separate multiple prompts using the `|` character, and get all combinations of them.")
				st.session_state["normalize_prompt_weights"] = st.checkbox("Normalize Prompt Weights.", 
					                                                   value=defaults.txt2vid.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
				st.session_state["save_individual_images"] = st.checkbox("Save individual images.",
					                                                 value=defaults.txt2vid.save_individual_images, help="Save each image generated before any filter or enhancement is applied.")
				st.session_state["save_video"] = st.checkbox("Save video",value=defaults.txt2vid.save_video, help="Save a video with all the images generated as frames at the end of the generation.")
				st.session_state["group_by_prompt"] = st.checkbox("Group results by prompt", value=defaults.txt2vid.group_by_prompt,
					                                          help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
				st.session_state["write_info_files"] = st.checkbox("Write Info file", value=defaults.txt2vid.write_info_files,
					                                           help="Save a file next to the image with informartion about the generation.")
				st.session_state["dynamic_preview_frequency"] = st.checkbox("Dynamic Preview Frequency", value=defaults.txt2vid.dynamic_preview_frequency,
					                                           help="This option tries to find the best value at which we can update \
					                                           the preview image during generation while minimizing the impact it has in performance. Default: True")
				st.session_state["do_loop"] = st.checkbox("Do Loop", value=defaults.txt2vid.do_loop,
					                                  help="Do loop")
				st.session_state["save_as_jpg"] = st.checkbox("Save samples as jpg", value=defaults.txt2vid.save_as_jpg, help="Saves the images as jpg instead of png.")
	
				if GFPGAN_available:
					st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=defaults.txt2vid.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation. This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
				else:
					st.session_state["use_GFPGAN"] = False
	
				if RealESRGAN_available:
					st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=defaults.txt2vid.use_RealESRGAN,
						                                         help="Uses the RealESRGAN model to upscale the images after the generation. This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")	
					st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
				else:
					st.session_state["use_RealESRGAN"] = False
					st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"
	
				st.session_state["variant_amount"] = st.slider("Variant Amount:", value=defaults.txt2vid.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
				st.session_state["variant_seed"] = st.text_input("Variant Seed:", value=defaults.txt2vid.seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")
				st.session_state["beta_start"] = st.slider("Beta Start:", value=defaults.txt2vid.beta_start, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")
				st.session_state["beta_end"] = st.slider("Beta End:", value=defaults.txt2vid.beta_end, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")
	
		if generate_button:
			#print("Loading models")
			# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.		
			#load_models(False, False, False, RealESRGAN_model, CustomModel_available=CustomModel_available, custom_model=custom_model)						
			
			# run video generation
			image, seed, info, stats = txt2vid(prompts=prompt, gpu=defaults.general.gpu,
				                   num_steps=st.session_state.sampling_steps, max_frames=int(st.session_state.max_frames),
				                   num_inference_steps=st.session_state.num_inference_steps,
				                   cfg_scale=cfg_scale,do_loop=st.session_state["do_loop"],
				                   seeds=seed, quality=100, eta=0.0, width=width,
				                   height=height, weights_path=custom_model, scheduler=scheduler_name,
				                   disable_tqdm=False, beta_start=st.session_state["beta_start"], beta_end=st.session_state["beta_end"],
				                   beta_schedule=beta_scheduler_type)
			    
			#message.success('Done!', icon="✅")
			message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")
	
			#except (StopException, KeyError):
				#print(f"Received Streamlit StopException")
	
			# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
			# use the current col2 first tab to show the preview_img and update it as its generated.
			#preview_image.image(output_images)		


#on import run init
def createHTMLGallery(images,info):
		html3 = """
		<div class="gallery-history" style="
	    display: flex;
	    flex-wrap: wrap;
		align-items: flex-start;">
		"""
		mkdwn_array = []
		for i in images:
			try:
				seed = info[images.index(i)]
			except:
				seed = ' '
			image_io = BytesIO()
			i.save(image_io, 'PNG')
			width, height = i.size
			#get random number for the id
			image_id = "%s" % (str(images.index(i)))
			(data, mimetype) = STImage._normalize_to_bytes(image_io.getvalue(), width, 'auto')
			this_file = in_memory_file_manager.add(data, mimetype, image_id)
			img_str = this_file.url
			#img_str = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
			#get image size
			
			#make sure the image is not bigger then 150px but keep the aspect ratio
			if width > 150:
				height = int(height * (150/width))
				width = 150
			if height > 150:
				width = int(width * (150/height))
				height = 150

			#mkdwn = f"""<img src="{img_str}" alt="Image" with="200" height="200" />"""
			mkdwn = f'''<div class="gallery" style="margin: 3px;" >
			<a href="{img_str}">
			  <img src="{img_str}" alt="Image" width="{width}" height="{height}">
			</a>
			<div class="desc" style="text-align: center; opacity: 40%;">{seed}</div>
		      </div>
		      '''
			mkdwn_array.append(mkdwn)
			
		html3 += "".join(mkdwn_array)
		html3 += '</div>'
		return html3
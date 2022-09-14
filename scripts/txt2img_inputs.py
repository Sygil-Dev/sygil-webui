from webui_streamlit import st
from sd_utils import *
from streamlit import StopException
import os
from io import BytesIO
from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
from streamlit.elements import image as STImage

from txt2img import txt2img

if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
	GFPGAN_available = True
else:
	GFPGAN_available = False

if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{defaults.general.RealESRGAN_model}.pth")):
	RealESRGAN_available = True
else:
	RealESRGAN_available = False	

def layout():
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
				
			with st.expander("Preview Settings"):
				st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=defaults.txt2img.update_preview,
			                                                         help="If enabled the image preview will be updated during the generation instead of at the end. \
			                                                         You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
			                                                         By default this is enabled and the frequency is set to 1 step.")
				
				st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=defaults.txt2img.update_preview_frequency,
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

				message = st.empty()
				
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
				normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=True, help="Ensure the sum of all weights add up to 1.0")
				save_individual_images = st.checkbox("Save individual images.", value=True, help="Save each image generated before any filter or enhancement is applied.")
				save_grid = st.checkbox("Save grid",value=True, help="Save a grid with all the images generated into a single image.")
				group_by_prompt = st.checkbox("Group results by prompt", value=True,
																	help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
				write_info_files = st.checkbox("Write Info file", value=True, help="Save a file next to the image with informartion about the generation.")
				save_as_jpg = st.checkbox("Save samples as jpg", value=False, help="Saves the images as jpg instead of png.")

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
		galleryCont = st.empty()

		if generate_button:
			#print("Loading models")
			# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.		
			load_models(False, use_GFPGAN, use_RealESRGAN, RealESRGAN_model)                

			try:
				output_images, seeds, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, RealESRGAN_model, batch_count, 1, 
										cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
										save_grid, group_by_prompt, save_as_jpg, use_GFPGAN, use_RealESRGAN, RealESRGAN_model, fp=defaults.general.fp,
										variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files)

				message.success('Done!', icon="✅")
				history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont = st.session_state['historyTab']

				if 'latestImages' in st.session_state:
					for i in output_images:
						#push the new image to the list of latest images and remove the oldest one
						#remove the last index from the list\
						st.session_state['latestImages'].pop()
						#add the new image to the start of the list
						st.session_state['latestImages'].insert(0, i)
					PlaceHolder.empty()
					with PlaceHolder.container():
						col1, col2, col3 = st.columns(3)
						col1_cont = st.container()
						col2_cont = st.container()
						col3_cont = st.container()
						with col1_cont:
							with col1:
								st.image(st.session_state['latestImages'][0])
								st.image(st.session_state['latestImages'][3])
								st.image(st.session_state['latestImages'][6])
						with col2_cont:
							with col2:
								st.image(st.session_state['latestImages'][1])
								st.image(st.session_state['latestImages'][4])
								st.image(st.session_state['latestImages'][7])
						with col3_cont:
							with col3:
								st.image(st.session_state['latestImages'][2])
								st.image(st.session_state['latestImages'][5])
								st.image(st.session_state['latestImages'][8])
						historyGallery = st.empty()
				
				# check if output_images length is the same as seeds length
				with gallery_tab:
					st.markdown(createHTMLGallery(output_images,seeds), unsafe_allow_html=True)
					
				
				st.session_state['historyTab'] = [history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont]
			except (StopException, KeyError):
				print(f"Received Streamlit StopException")

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
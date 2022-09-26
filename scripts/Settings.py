# base webui import and utils.
from webui_streamlit import st
from sd_utils import *

# streamlit imports

#streamlit components section
import streamlit_nested_layout
from streamlit_server_state import server_state, server_state_lock

#other imports
from omegaconf import OmegaConf

# end of imports
#---------------------------------------------------------------------------------------------------------------

def layout():		
	st.header("Settings")
	
	with st.form("Settings"):
		general_tab, txt2img_tab, img2img_tab, \
			txt2vid_tab, textual_inversion_tab, concepts_library_tab = st.tabs(['General', "Text-To-Image",
																				"Image-To-Image", "Text-To-Video",
																				"Textual Inversion",	
																				"Concepts Library"])
		
		with general_tab:
			col1, col2, col3, col4, col5 = st.columns(5, gap='large')
			
			device_list = []
			device_properties = [(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]
			for device in device_properties:
					id = device[0]
					name = device[1].name
					total_memory = device[1].total_memory
					
					device_list.append(f"{id}: {name} ({human_readable_size(total_memory, decimal_places=0)})")
					
			
			with col1:
				st.title("General")	
				st.session_state['defaults'].general.gpu = int(st.selectbox("GPU", device_list,
																		help=f"Select which GPU to use. Default: {device_list[0]}").split(":")[0])
				
				st.session_state['defaults'].general.outdir = str(st.text_input("Output directory", value=st.session_state['defaults'].general.outdir,
																			help="Relative directory on which the output images after a generation will be placed. Default: 'outputs'"))
				
				# If we have custom models available on the "models/custom" 
				# folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
				custom_models_available()
				
				if st.session_state.CustomModel_available:
					st.session_state.default_model = st.selectbox("Default Model:", server_state["custom_models"],
																		 index=server_state["custom_models"].index(st.session_state['defaults'].general.default_model),
																			help="Select the model you want to use. If you have placed custom models \
																			on your 'models/custom' folder they will be shown here as well. The model name that will be shown here \
																			is the same as the name the file for the model has on said folder, \
																			it is recommended to give the .ckpt file a name that \
																			will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
				else:
						st.session_state.default_model = st.selectbox("Default Model:", [st.session_state['defaults'].general.default_model],
																		 help="Select the model you want to use. If you have placed custom models \
																			on your 'models/custom' folder they will be shown here as well. \
																			The model name that will be shown here is the same as the name\
																			the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
																			will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
					
				st.session_state['defaults'].general.default_model_config = st.text_input("Default Model Config", value=st.session_state['defaults'].general.default_model_config,
																			help="Default model config file for inference. Default: 'configs/stable-diffusion/v1-inference.yaml'")
				
				st.session_state['defaults'].general.default_model_path = st.text_input("Default Model Config", value=st.session_state['defaults'].general.default_model_path,
																			help="Default model path. Default: 'models/ldm/stable-diffusion-v1/model.ckpt'")		
						
				st.session_state['defaults'].general.GFPGAN_dir = st.text_input("Default GFPGAN directory", value=st.session_state['defaults'].general.GFPGAN_dir,
																			help="Default GFPGAN directory. Default: './src/gfpgan'")
				
				st.session_state['defaults'].general.RealESRGAN_dir = st.text_input("Default RealESRGAN directory", value=st.session_state['defaults'].general.RealESRGAN_dir,
																			help="Default GFPGAN directory. Default: './src/realesrgan'")
				
				RealESRGAN_model_list = ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"]
				st.session_state['defaults'].general.RealESRGAN_model = st.selectbox("RealESRGAN model", RealESRGAN_model_list,
																					 index=RealESRGAN_model_list.index(st.session_state['defaults'].general.RealESRGAN_model),
																					 help="Default RealESRGAN model. Default: 'RealESRGAN_x4plus'")
				

			with col2:
				st.title("Performance")	
				
				st.session_state["defaults"].general.gfpgan_cpu = st.checkbox("GFPGAN - CPU", value=st.session_state['defaults'].general.gfpgan_cpu,
																			  help="Run GFPGAN on the cpu. Default: False")
				
				st.session_state["defaults"].general.esrgan_cpu = st.checkbox("ESRGAN - CPU", value=st.session_state['defaults'].general.esrgan_cpu,
																			  help="Run ESRGAN on the cpu. Default: False")
				
				st.session_state["defaults"].general.extra_models_cpu = st.checkbox("Extra Models - CPU", value=st.session_state['defaults'].general.extra_models_cpu,
																					help="Run extra models (GFGPAN/ESRGAN) on cpu. Default: False")
				
				st.session_state["defaults"].general.extra_models_gpu = st.checkbox("Extra Models - GPU", value=st.session_state['defaults'].general.extra_models_gpu,
																					help="Run extra models (GFGPAN/ESRGAN) on gpu. \
																					Check and save in order to be able to select the GPU that each model will use. Default: False")
				if st.session_state["defaults"].general.extra_models_gpu:
					st.session_state['defaults'].general.gfpgan_gpu = int(st.selectbox("GFGPAN GPU", device_list, index=st.session_state['defaults'].general.gfpgan_gpu,
																				   help=f"Select which GPU to use. Default: {device_list[st.session_state['defaults'].general.gfpgan_gpu]}",
																				   key="gfpgan_gpu").split(":")[0])
					
					st.session_state["defaults"].general.esrgan_gpu = int(st.selectbox("ESRGAN - GPU", device_list, index=st.session_state['defaults'].general.esrgan_gpu,
																						help=f"Select which GPU to use. Default: {device_list[st.session_state['defaults'].general.esrgan_gpu]}",
																						key="esrgan_gpu").split(":")[0])
					
				st.session_state["defaults"].general.no_half = st.checkbox("No Half", value=st.session_state['defaults'].general.no_half,
																		   help="DO NOT switch the model to 16-bit floats. Default: False")
				
				st.session_state["defaults"].general.use_float16 = st.checkbox("Use float16", value=st.session_state['defaults'].general.use_float16,
																		   help="Switch the model to 16-bit floats. Default: False")
				
				precision_list = ['full','autocast']
				st.session_state["defaults"].general.precision = st.selectbox("Precision", precision_list, index=precision_list.index(st.session_state['defaults'].general.precision),
																			  help="Evaluates at this precision. Default: autocast")
				
				st.session_state["defaults"].general.optimized = st.checkbox("Optimized Mode", value=st.session_state['defaults'].general.optimized,
																		   help="Loads the model onto the device piecemeal instead of all at once to reduce VRAM usage\
																		   at the cost of performance. Default: False")
				
				st.session_state["defaults"].general.optimized_turbo = st.checkbox("Optimized Turbo Mode", value=st.session_state['defaults'].general.optimized_turbo,
																						   help="Alternative optimization mode that does not save as much VRAM but \
																						   runs siginificantly faster. Default: False")
				
				st.session_state["defaults"].general.optimized_config = st.text_input("Optimized Config", value=st.session_state['defaults'].general.optimized_config,
																					help=f"Loads alternative optimized configuration for inference. \
																					Default: optimizedSD/v1-inference.yaml")
				
				st.session_state["defaults"].general.enable_attention_slicing = st.checkbox("Enable Attention Slicing", value=st.session_state['defaults'].general.enable_attention_slicing,
																						   help="Enable sliced attention computation. When this option is enabled, the attention module will \
																						   split the input tensor in slices, to compute attention in several steps. This is useful to save some \
																						   memory in exchange for a small speed decrease. Only works the txt2vid tab right now. Default: False")
				
				st.session_state["defaults"].general.enable_minimal_memory_usage = st.checkbox("Enable Minimal Memory Usage", value=st.session_state['defaults'].general.enable_minimal_memory_usage,
																							help="Moves only unet to fp16 and to CUDA, while keepping lighter models on CPUs \
																							(Not properly implemented and currently not working, check this \
																							link 'https://github.com/huggingface/diffusers/pull/537' for more information on it ). Default: False")	
				
				st.session_state["defaults"].general.update_preview = st.checkbox("Update Preview Image", value=st.session_state['defaults'].general.update_preview,
																					help="Enables the preview image to be updated and shown to the user on the UI during the generation.\
																					If checked, once you save the settings an option to specify the frequency at which the image is updated\
																					in steps will be shown, this is helpful to reduce the negative effect this option has on performance. Default: True")				
				if st.session_state["defaults"].general.update_preview:
					st.session_state["defaults"].general.update_preview_frequency = int(st.text_input("Update Preview Frequency", value=st.session_state['defaults'].general.update_preview_frequency,
																						help="Specify the frequency at which the image is updated in steps, this is helpful to reduce the \
																						negative effect updating the preview image has on performance. Default: 10"))					
				
			with col3:
				st.session_state["defaults"].general.use_sd_concepts_library = st.checkbox("Use the Concepts Library", value=st.session_state['defaults'].general.use_sd_concepts_library,
																									   help="Use the embeds Concepts Library, if checked, once the settings are saved an option will\
																									  appear to specify the directory where the concepts are stored. Default: True)")				
			
				if st.session_state["defaults"].general.use_sd_concepts_library:
					st.session_state['defaults'].general.sd_concepts_library_folder = st.text_input("Concepts Library Folder",
																									value=st.session_state['defaults'].general.sd_concepts_library_folder,
																									help="Relative folder on which the concepts library embeds are stored. \
																									Default: 'models/custom/sd-concepts-library'")	
					
				st.session_state['defaults'].general.LDSR_dir = st.text_input("LDSR Folder", value=st.session_state['defaults'].general.LDSR_dir,
																			  help="Folder where LDSR is located. Default: './src/latent-diffusion'")
				
				st.session_state["defaults"].general.save_metadata = st.checkbox("Save Metadata", value=st.session_state['defaults'].general.save_metadata,
																									   help="Save metadata on the output image. Default: True")
				save_format_list = ["png"]
				st.session_state["defaults"].general.save_format = st.selectbox("Save Format",save_format_list, index=save_format_list.index(st.session_state['defaults'].general.save_format),
																									   help="Format that will be used whens saving the output images. Default: 'png'")
				
				st.session_state["defaults"].general.skip_grid = st.checkbox("Skip Grid", value=st.session_state['defaults'].general.skip_grid,
																									   help="Skip saving the grid output image. Default: False")
				if not st.session_state["defaults"].general.skip_grid:
					st.session_state["defaults"].general.grid_format = st.text_input("Grid Format", value=st.session_state['defaults'].general.grid_format,
																					 help="Format for saving the grid output image. Default: 'jpg:95'")
				
				st.session_state["defaults"].general.skip_save = st.checkbox("Skip Save", value=st.session_state['defaults'].general.skip_save,
																									   help="Skip saving the output image. Default: False")
				
				st.session_state["defaults"].general.n_rows = int(st.text_input("Number of Grid Rows", value=st.session_state['defaults'].general.n_rows,
																				 help="Number of rows the grid wil have when saving the grid output image. Default: '-1'"))
			
				st.session_state["defaults"].general.no_verify_input = st.checkbox("Do not Verify Input", value=st.session_state['defaults'].general.no_verify_input,
																									   help="Do not verify input to check if it's too long. Default: False")
				
				
		with txt2img_tab:
			st.title("Text To Image")
			st.info("Under Construction. :construction_worker:")
			
		with img2img_tab:
			st.title("Image To Image")
			st.info("Under Construction. :construction_worker:")
			
		with txt2vid_tab:
			st.title("Text To Video")
			st.info("Under Construction. :construction_worker:")
			
		with textual_inversion_tab:
			st.title("Textual Inversion")
			st.info("Under Construction. :construction_worker:")
		
		with concepts_library_tab:
			st.title("Concepts Library")
			#st.info("Under Construction. :construction_worker:")	
			col1, col2, col3, col4, col5 = st.columns(5, gap='large')
			with col1:
				st.session_state["defaults"].concepts_library.concepts_per_page = int(st.text_input("Concepts Per Page", value=st.session_state['defaults'].concepts_library.concepts_per_page,
																									help="Number of concepts per page to show on the Concepts Library. Default: '12'"))
		
		# add space for the buttons at the bottom	
		st.markdown("---")
		
		# We need a submit button to save the Settings
		# as well as one to reset them to the defaults, just in case.
		_, _, save_button_col, reset_button_col, _, _ = st.columns([1,1,1,1,1,1], gap="large")
		with save_button_col:
			save_button = st.form_submit_button("Save")
			
		with reset_button_col:
			reset_button = st.form_submit_button("Reset")
			
		if save_button:
			#userconfig_streamlit = OmegaConf.to_yaml(st.session_state.defaults)
			#OmegaConf.save("configs/webui/userconfig_streamlit.yaml")
					
			OmegaConf.save(config=st.session_state.defaults, f="configs/webui/userconfig_streamlit.yaml")
			loaded = OmegaConf.load("configs/webui/userconfig_streamlit.yaml")
			assert st.session_state.defaults == loaded			
			
		if reset_button:
			st.session_state["defaults"] = OmegaConf.load("configs/webui/webui_streamlit.yaml")
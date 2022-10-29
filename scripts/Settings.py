# This file is part of sygil-webui (https://github.com/Sygil-Dev/sygil-webui/).

# Copyright 2022 Sygil-Dev team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# base webui import and utils.
from sd_utils import *

# streamlit imports

# streamlit components section
import streamlit_nested_layout
from streamlit_server_state import server_state, server_state_lock

# other imports
from omegaconf import OmegaConf

# end of imports
# ---------------------------------------------------------------------------------------------------------------

@logger.catch(reraise=True)
def layout():
    #st.header("Settings")

    with st.form("Settings"):
        general_tab, txt2img_tab, img2img_tab, img2txt_tab, txt2vid_tab, image_processing, textual_inversion_tab, concepts_library_tab = st.tabs(
            ['General', "Text-To-Image", "Image-To-Image", "Image-To-Text", "Text-To-Video", "Image processing", "Textual Inversion", "Concepts Library"])

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

                if server_state["CustomModel_available"]:
                    st.session_state.defaults.general.default_model = st.selectbox("Default Model:", server_state["custom_models"],
                                                                  index=server_state["custom_models"].index(st.session_state['defaults'].general.default_model),
                                                                  help="Select the model you want to use. If you have placed custom models \
                                                                  on your 'models/custom' folder they will be shown here as well. The model name that will be shown here \
                                                                  is the same as the name the file for the model has on said folder, \
                                                                  it is recommended to give the .ckpt file a name that \
                                                                  will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
                else:
                    st.session_state.defaults.general.default_model = st.selectbox("Default Model:", [st.session_state['defaults'].general.default_model],
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
                                                                                help="Default GFPGAN directory. Default: './models/gfpgan'")

                st.session_state['defaults'].general.RealESRGAN_dir = st.text_input("Default RealESRGAN directory", value=st.session_state['defaults'].general.RealESRGAN_dir,
                                                                                    help="Default GFPGAN directory. Default: './models/realesrgan'")

                RealESRGAN_model_list = ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"]
                st.session_state['defaults'].general.RealESRGAN_model = st.selectbox("RealESRGAN model", RealESRGAN_model_list,
                                                                                     index=RealESRGAN_model_list.index(st.session_state['defaults'].general.RealESRGAN_model),
                                                                                     help="Default RealESRGAN model. Default: 'RealESRGAN_x4plus'")
                Upscaler_list = ["RealESRGAN", "LDSR"]
                st.session_state['defaults'].general.upscaling_method = st.selectbox("Upscaler", Upscaler_list, index=Upscaler_list.index(
                    st.session_state['defaults'].general.upscaling_method), help="Default upscaling method. Default: 'RealESRGAN'")

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

                precision_list = ['full', 'autocast']
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

                # st.session_state["defaults"].general.update_preview = st.checkbox("Update Preview Image", value=st.session_state['defaults'].general.update_preview,
                # help="Enables the preview image to be updated and shown to the user on the UI during the generation.\
                # If checked, once you save the settings an option to specify the frequency at which the image is updated\
                # in steps will be shown, this is helpful to reduce the negative effect this option has on performance. \
                # Default: True")
                st.session_state["defaults"].general.update_preview = True
                st.session_state["defaults"].general.update_preview_frequency = st.number_input("Update Preview Frequency",
                                                                                                min_value=0,
                                                                                                value=st.session_state['defaults'].general.update_preview_frequency,
                                                                                                help="Specify the frequency at which the image is updated in steps, this is helpful to reduce the \
                                                                                                negative effect updating the preview image has on performance. Default: 10")

            with col3:
                st.title("Others")
                st.session_state["defaults"].general.use_sd_concepts_library = st.checkbox("Use the Concepts Library", value=st.session_state['defaults'].general.use_sd_concepts_library,
                                                                                           help="Use the embeds Concepts Library, if checked, once the settings are saved an option will\
                                                                                           appear to specify the directory where the concepts are stored. Default: True)")

                if st.session_state["defaults"].general.use_sd_concepts_library:
                    st.session_state['defaults'].general.sd_concepts_library_folder = st.text_input("Concepts Library Folder",
                                                                                                    value=st.session_state['defaults'].general.sd_concepts_library_folder,
                                                                                                    help="Relative folder on which the concepts library embeds are stored. \
                                                                                                    Default: 'models/custom/sd-concepts-library'")

                st.session_state['defaults'].general.LDSR_dir = st.text_input("LDSR Folder", value=st.session_state['defaults'].general.LDSR_dir,
                                                                              help="Folder where LDSR is located. Default: './models/ldsr'")

                st.session_state["defaults"].general.save_metadata = st.checkbox("Save Metadata", value=st.session_state['defaults'].general.save_metadata,
                                                                                 help="Save metadata on the output image. Default: True")
                save_format_list = ["png","jpg", "jpeg","webp"]
                st.session_state["defaults"].general.save_format = st.selectbox("Save Format", save_format_list, index=save_format_list.index(st.session_state['defaults'].general.save_format),
                                                                                help="Format that will be used whens saving the output images. Default: 'png'")

                st.session_state["defaults"].general.skip_grid = st.checkbox("Skip Grid", value=st.session_state['defaults'].general.skip_grid,
                                                                             help="Skip saving the grid output image. Default: False")
                if not st.session_state["defaults"].general.skip_grid:


                    st.session_state["defaults"].general.grid_quality = st.number_input("Grid Quality", value=st.session_state['defaults'].general.grid_quality,
                                                                                        help="Format for saving the grid output image. Default: 95")

                st.session_state["defaults"].general.skip_save = st.checkbox("Skip Save", value=st.session_state['defaults'].general.skip_save,
                                                                             help="Skip saving the output image. Default: False")

                st.session_state["defaults"].general.n_rows = st.number_input("Number of Grid Rows", value=st.session_state['defaults'].general.n_rows,
                                                                              help="Number of rows the grid wil have when saving the grid output image. Default: '-1'")

                st.session_state["defaults"].general.no_verify_input = st.checkbox("Do not Verify Input", value=st.session_state['defaults'].general.no_verify_input,
                                                                                   help="Do not verify input to check if it's too long. Default: False")

                st.session_state["defaults"].daisi_app.running_on_daisi_io = st.checkbox("Running on Daisi.io?", value=st.session_state['defaults'].daisi_app.running_on_daisi_io,
                                                                                         help="Specify if we are running on app.Daisi.io . Default: False")

            with col4:
                st.title("Streamlit Config")

                default_theme_list = ["light", "dark"]
                st.session_state["defaults"].general.default_theme = st.selectbox("Default Theme", default_theme_list, index=default_theme_list.index(st.session_state['defaults'].general.default_theme),
                                                                                  help="Defaut theme to use as base for streamlit. Default: dark")
                st.session_state["streamlit_config"]["theme"]["base"] = st.session_state["defaults"].general.default_theme


                if not st.session_state['defaults'].admin.hide_server_setting:
                    with st.expander("Server", True):

                        st.session_state["streamlit_config"]['server']['headless'] = st.checkbox("Run Headless", help="If false, will attempt to open a browser window on start.  \
                                                                                                 Default: false unless (1) we are on a Linux box where DISPLAY is unset, \
                                                                                                 or (2) we are running in the Streamlit Atom plugin.")

                        st.session_state["streamlit_config"]['server']['port'] = st.number_input("Port", value=st.session_state["streamlit_config"]['server']['port'],
                                                                                                 help="The port where the server will listen for browser connections. Default: 8501")

                        st.session_state["streamlit_config"]['server']['baseUrlPath'] = st.text_input("Base Url Path", value=st.session_state["streamlit_config"]['server']['baseUrlPath'],
                                                                                                 help="The base path for the URL where Streamlit should be served from. Default: '' ")

                        st.session_state["streamlit_config"]['server']['enableCORS'] = st.checkbox("Enable CORS", value=st.session_state['streamlit_config']['server']['enableCORS'],
                                                                                                   help="Enables support for Cross-Origin Request Sharing (CORS) protection, for added security. \
                                                                                                   Due to conflicts between CORS and XSRF, if `server.enableXsrfProtection` is on and `server.enableCORS` \
                                                                                                   is off at the same time, we will prioritize `server.enableXsrfProtection`. Default: true")

                        st.session_state["streamlit_config"]['server']['enableXsrfProtection'] = st.checkbox("Enable Xsrf Protection",
                                                                                                             value=st.session_state['streamlit_config']['server']['enableXsrfProtection'],
                                                                                                             help="Enables support for Cross-Site Request Forgery (XSRF) protection, \
                                                                                                             for added security. Due to conflicts between CORS and XSRF, \
                                                                                                             if `server.enableXsrfProtection` is on and `server.enableCORS` is off at \
                                                                                                             the same time, we will prioritize `server.enableXsrfProtection`. Default: true")

                        st.session_state["streamlit_config"]['server']['maxUploadSize'] = st.number_input("Max Upload Size", value=st.session_state["streamlit_config"]['server']['maxUploadSize'],
                                                                                                 help="Max size, in megabytes, for files uploaded with the file_uploader. Default: 200")

                        st.session_state["streamlit_config"]['server']['maxMessageSize'] = st.number_input("Max Message Size", value=st.session_state["streamlit_config"]['server']['maxUploadSize'],
                                                                                                 help="Max size, in megabytes, of messages that can be sent via the WebSocket connection. Default: 200")

                        st.session_state["streamlit_config"]['server']['enableWebsocketCompression'] = st.checkbox("Enable Websocket Compression",
                                                                                                                   value=st.session_state["streamlit_config"]['server']['enableWebsocketCompression'],
                                                                                                                   help=" Enables support for websocket compression. Default: false")
                if not st.session_state['defaults'].admin.hide_browser_setting:
                    with st.expander("Browser", expanded=True):
                        st.session_state["streamlit_config"]['browser']['serverAddress'] = st.text_input("Server Address",
                                                                                                       value=st.session_state["streamlit_config"]['browser']['serverAddress'] if "serverAddress" in st.session_state["streamlit_config"] else "localhost",
                                                                                                       help="Internet address where users should point their browsers in order \
                                                                                                       to connect to the app. Can be IP address or DNS name and path.\
                                                                                                       This is used to: - Set the correct URL for CORS and XSRF protection purposes. \
                                                                                                       - Show the URL on the terminal - Open the browser. Default: 'localhost'")

                        st.session_state["defaults"].general.streamlit_telemetry = st.checkbox("Enable Telemetry", value=st.session_state['defaults'].general.streamlit_telemetry,
                                                                                               help="Enables or Disables streamlit telemetry. Default: False")
                        st.session_state["streamlit_config"]["browser"]["gatherUsageStats"] = st.session_state["defaults"].general.streamlit_telemetry

                        st.session_state["streamlit_config"]['browser']['serverPort'] = st.number_input("Server Port", value=st.session_state["streamlit_config"]['browser']['serverPort'],
                                                                                                 help="Port where users should point their browsers in order to connect to the app. \
                                                                                                 This is used to: - Set the correct URL for CORS and XSRF protection purposes. \
                                                                                                 - Show the URL on the terminal - Open the browser \
                                                                                                 Default: whatever value is set in server.port.")

            with col5:
                st.title("Huggingface")
                st.session_state["defaults"].general.huggingface_token = st.text_input("Huggingface Token", value=st.session_state['defaults'].general.huggingface_token, type="password",
                                                                                       help="Your Huggingface Token, it's used to download the model for the diffusers library which \
                                                                                       is used on the Text To Video tab. This token will be saved to your user config file\
                                                                                       and WILL NOT be share with us or anyone. You can get your access token \
                                                                                       at https://huggingface.co/settings/tokens. Default: None")

        with txt2img_tab:
            col1, col2, col3, col4, col5 = st.columns(5, gap='medium')

            with col1:
                st.title("Slider Parameters")

                # Width
                st.session_state["defaults"].txt2img.width.value = st.number_input("Default Image Width", value=st.session_state['defaults'].txt2img.width.value,
                                                                                   help="Set the default width for the generated image. Default is: 512")

                st.session_state["defaults"].txt2img.width.min_value = st.number_input("Minimum Image Width", value=st.session_state['defaults'].txt2img.width.min_value,
                                                                                       help="Set the default minimum value for the width slider. Default is: 64")

                st.session_state["defaults"].txt2img.width.max_value = st.number_input("Maximum Image Width", value=st.session_state['defaults'].txt2img.width.max_value,
                                                                                       help="Set the default maximum value for the width slider. Default is: 2048")

                # Height
                st.session_state["defaults"].txt2img.height.value = st.number_input("Default Image Height", value=st.session_state['defaults'].txt2img.height.value,
                                                                                    help="Set the default height for the generated image. Default is: 512")

                st.session_state["defaults"].txt2img.height.min_value = st.number_input("Minimum Image Height", value=st.session_state['defaults'].txt2img.height.min_value,
                                                                                        help="Set the default minimum value for the height slider. Default is: 64")

                st.session_state["defaults"].txt2img.height.max_value = st.number_input("Maximum Image Height", value=st.session_state['defaults'].txt2img.height.max_value,
                                                                                        help="Set the default maximum value for the height slider. Default is: 2048")

                with col2:
                    # CFG
                    st.session_state["defaults"].txt2img.cfg_scale.value = st.number_input("Default CFG Scale", value=st.session_state['defaults'].txt2img.cfg_scale.value,
                                                                                           help="Set the default value for the CFG Scale. Default is: 7.5")

                    st.session_state["defaults"].txt2img.cfg_scale.min_value = st.number_input("Minimum CFG Scale Value", value=st.session_state['defaults'].txt2img.cfg_scale.min_value,
                                                                                               help="Set the default minimum value for the CFG scale slider. Default is: 1")

                    st.session_state["defaults"].txt2img.cfg_scale.step = st.number_input("CFG Slider Steps", value=st.session_state['defaults'].txt2img.cfg_scale.step,
                                                                                          help="Set the default value for the number of steps on the CFG scale slider. Default is: 0.5")
                    # Sampling Steps
                    st.session_state["defaults"].txt2img.sampling_steps.value = st.number_input("Default Sampling Steps", value=st.session_state['defaults'].txt2img.sampling_steps.value,
                                                                                                help="Set the default number of sampling steps to use. Default is: 30 (with k_euler)")

                    st.session_state["defaults"].txt2img.sampling_steps.min_value = st.number_input("Minimum Sampling Steps",
                                                                                                    value=st.session_state['defaults'].txt2img.sampling_steps.min_value,
                                                                                                    help="Set the default minimum value for the sampling steps slider. Default is: 1")

                    st.session_state["defaults"].txt2img.sampling_steps.step = st.number_input("Sampling Slider Steps",
                                                                                               value=st.session_state['defaults'].txt2img.sampling_steps.step,
                                                                                               help="Set the default value for the number of steps on the sampling steps slider. Default is: 10")

            with col3:
                st.title("General Parameters")

                # Batch Count
                st.session_state["defaults"].txt2img.batch_count.value = st.number_input("Batch count", value=st.session_state['defaults'].txt2img.batch_count.value,
                                                                                         help="How many iterations or batches of images to generate in total.")

                st.session_state["defaults"].txt2img.batch_size.value = st.number_input("Batch size", value=st.session_state.defaults.txt2img.batch_size.value,
                                                                                        help="How many images are at once in a batch.\
                                                                                        It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it \
                                                                                        takes to finish generation as more images are generated at once.\
                                                                                        Default: 1")

                default_sampler_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", "k_heun", "PLMS", "DDIM"]
                st.session_state["defaults"].txt2img.default_sampler = st.selectbox("Default Sampler",
                                                                                    default_sampler_list, index=default_sampler_list.index(
                                                                                        st.session_state['defaults'].txt2img.default_sampler),
                                                                                    help="Defaut sampler to use for txt2img. Default: k_euler")

                st.session_state['defaults'].txt2img.seed = st.text_input("Default Seed", value=st.session_state['defaults'].txt2img.seed, help="Default seed.")

            with col4:

                st.session_state["defaults"].txt2img.separate_prompts = st.checkbox("Separate Prompts",
                                                                                    value=st.session_state['defaults'].txt2img.separate_prompts, help="Separate Prompts. Default: False")

                st.session_state["defaults"].txt2img.normalize_prompt_weights = st.checkbox("Normalize Prompt Weights",
                                                                                            value=st.session_state['defaults'].txt2img.normalize_prompt_weights,
                                                                                            help="Choose to normalize prompt weights. Default: True")

                st.session_state["defaults"].txt2img.save_individual_images = st.checkbox("Save Individual Images",
                                                                                          value=st.session_state['defaults'].txt2img.save_individual_images,
                                                                                          help="Choose to save individual images. Default: True")

                st.session_state["defaults"].txt2img.save_grid = st.checkbox("Save Grid Images", value=st.session_state['defaults'].txt2img.save_grid,
                                                                             help="Choose to save the grid images. Default: True")

                st.session_state["defaults"].txt2img.group_by_prompt = st.checkbox("Group By Prompt", value=st.session_state['defaults'].txt2img.group_by_prompt,
                                                                                   help="Choose to save images grouped by their prompt. Default: False")

                st.session_state["defaults"].txt2img.save_as_jpg = st.checkbox("Save As JPG", value=st.session_state['defaults'].txt2img.save_as_jpg,
                                                                               help="Choose to save images as jpegs. Default: False")

                st.session_state["defaults"].txt2img.write_info_files = st.checkbox("Write Info Files For Images", value=st.session_state['defaults'].txt2img.write_info_files,
                                                                                    help="Choose to write the info files along with the generated images. Default: True")

                st.session_state["defaults"].txt2img.use_GFPGAN = st.checkbox(
                    "Use GFPGAN", value=st.session_state['defaults'].txt2img.use_GFPGAN, help="Choose to use GFPGAN. Default: False")

                st.session_state["defaults"].txt2img.use_upscaling = st.checkbox("Use Upscaling", value=st.session_state['defaults'].txt2img.use_upscaling,
                                                                                 help="Choose to turn on upscaling by default. Default: False")

                st.session_state["defaults"].txt2img.update_preview = True
                st.session_state["defaults"].txt2img.update_preview_frequency = st.number_input("Preview Image Update Frequency",
                                                                                                min_value=0,
                                                                                                value=st.session_state['defaults'].txt2img.update_preview_frequency,
                                                                                                help="Set the default value for the frrquency of the preview image updates. Default is: 10")

            with col5:
                st.title("Variation Parameters")

                st.session_state["defaults"].txt2img.variant_amount.value = st.number_input("Default Variation Amount",
                                                                                            value=st.session_state['defaults'].txt2img.variant_amount.value,
                                                                                            help="Set the default variation to use. Default is: 0.0")

                st.session_state["defaults"].txt2img.variant_amount.min_value = st.number_input("Minimum Variation Amount",
                                                                                                value=st.session_state['defaults'].txt2img.variant_amount.min_value,
                                                                                                help="Set the default minimum value for the variation slider. Default is: 0.0")

                st.session_state["defaults"].txt2img.variant_amount.max_value = st.number_input("Maximum Variation Amount",
                                                                                                value=st.session_state['defaults'].txt2img.variant_amount.max_value,
                                                                                                help="Set the default maximum value for the variation slider. Default is: 1.0")

                st.session_state["defaults"].txt2img.variant_amount.step = st.number_input("Variation Slider Steps",
                                                                                           value=st.session_state['defaults'].txt2img.variant_amount.step,
                                                                                           help="Set the default value for the number of steps on the variation slider. Default is: 1")

                st.session_state['defaults'].txt2img.variant_seed = st.text_input("Default Variation Seed", value=st.session_state['defaults'].txt2img.variant_seed,
                                                                                  help="Default variation seed.")

        with img2img_tab:
            col1, col2, col3, col4, col5 = st.columns(5, gap='medium')

            with col1:
                st.title("Image Editing")

                # Denoising
                st.session_state["defaults"].img2img.denoising_strength.value = st.number_input("Default Denoising Amount",
                                                                                                value=st.session_state['defaults'].img2img.denoising_strength.value,
                                                                                                help="Set the default denoising to use. Default is: 0.75")

                st.session_state["defaults"].img2img.denoising_strength.min_value = st.number_input("Minimum Denoising Amount",
                                                                                                    value=st.session_state['defaults'].img2img.denoising_strength.min_value,
                                                                                                    help="Set the default minimum value for the denoising slider. Default is: 0.0")

                st.session_state["defaults"].img2img.denoising_strength.max_value = st.number_input("Maximum Denoising Amount",
                                                                                                    value=st.session_state['defaults'].img2img.denoising_strength.max_value,
                                                                                                    help="Set the default maximum value for the denoising slider. Default is: 1.0")

                st.session_state["defaults"].img2img.denoising_strength.step = st.number_input("Denoising Slider Steps",
                                                                                               value=st.session_state['defaults'].img2img.denoising_strength.step,
                                                                                               help="Set the default value for the number of steps on the denoising slider. Default is: 0.01")

                # Masking
                st.session_state["defaults"].img2img.mask_mode = st.number_input("Default Mask Mode", value=st.session_state['defaults'].img2img.mask_mode,
                                                                                 help="Set the default mask mode to use. 0 = Keep Masked Area, 1 = Regenerate Masked Area. Default is: 0")

                st.session_state["defaults"].img2img.mask_restore = st.checkbox("Default Mask Restore", value=st.session_state['defaults'].img2img.mask_restore,
                                                                                help="Mask Restore. Default: False")

                st.session_state["defaults"].img2img.resize_mode = st.number_input("Default Resize Mode", value=st.session_state['defaults'].img2img.resize_mode,
                                                                                   help="Set the default resizing mode. 0 = Just Resize, 1 = Crop and Resize, 3 = Resize and Fill. Default is: 0")

            with col2:
                st.title("Slider Parameters")

                # Width
                st.session_state["defaults"].img2img.width.value = st.number_input("Default Outputted Image Width", value=st.session_state['defaults'].img2img.width.value,
                                                                                   help="Set the default width for the generated image. Default is: 512")

                st.session_state["defaults"].img2img.width.min_value = st.number_input("Minimum Outputted Image Width", value=st.session_state['defaults'].img2img.width.min_value,
                                                                                       help="Set the default minimum value for the width slider. Default is: 64")

                st.session_state["defaults"].img2img.width.max_value = st.number_input("Maximum Outputted Image Width", value=st.session_state['defaults'].img2img.width.max_value,
                                                                                       help="Set the default maximum value for the width slider. Default is: 2048")

                # Height
                st.session_state["defaults"].img2img.height.value = st.number_input("Default Outputted Image Height", value=st.session_state['defaults'].img2img.height.value,
                                                                                    help="Set the default height for the generated image. Default is: 512")

                st.session_state["defaults"].img2img.height.min_value = st.number_input("Minimum Outputted Image Height", value=st.session_state['defaults'].img2img.height.min_value,
                                                                                        help="Set the default minimum value for the height slider. Default is: 64")

                st.session_state["defaults"].img2img.height.max_value = st.number_input("Maximum Outputted Image Height", value=st.session_state['defaults'].img2img.height.max_value,
                                                                                        help="Set the default maximum value for the height slider. Default is: 2048")

                # CFG
                st.session_state["defaults"].img2img.cfg_scale.value = st.number_input("Default Img2Img CFG Scale", value=st.session_state['defaults'].img2img.cfg_scale.value,
                                                                                       help="Set the default value for the CFG Scale. Default is: 7.5")

                st.session_state["defaults"].img2img.cfg_scale.min_value = st.number_input("Minimum Img2Img CFG Scale Value",
                                                                                           value=st.session_state['defaults'].img2img.cfg_scale.min_value,
                                                                                           help="Set the default minimum value for the CFG scale slider. Default is: 1")

                with col3:
                    st.session_state["defaults"].img2img.cfg_scale.step = st.number_input("Img2Img CFG Slider Steps",
                                                                                          value=st.session_state['defaults'].img2img.cfg_scale.step,
                                                                                          help="Set the default value for the number of steps on the CFG scale slider. Default is: 0.5")

                    # Sampling Steps
                    st.session_state["defaults"].img2img.sampling_steps.value = st.number_input("Default Img2Img Sampling Steps",
                                                                                                value=st.session_state['defaults'].img2img.sampling_steps.value,
                                                                                                help="Set the default number of sampling steps to use. Default is: 30 (with k_euler)")

                    st.session_state["defaults"].img2img.sampling_steps.min_value = st.number_input("Minimum Img2Img Sampling Steps",
                                                                                                    value=st.session_state['defaults'].img2img.sampling_steps.min_value,
                                                                                                    help="Set the default minimum value for the sampling steps slider. Default is: 1")

                    st.session_state["defaults"].img2img.sampling_steps.step = st.number_input("Img2Img Sampling Slider Steps",
                                                                                               value=st.session_state['defaults'].img2img.sampling_steps.step,
                                                                                               help="Set the default value for the number of steps on the sampling steps slider. Default is: 10")

                    # Batch Count
                    st.session_state["defaults"].img2img.batch_count.value = st.number_input("Img2img Batch count", value=st.session_state["defaults"].img2img.batch_count.value,
                                                                                             help="How many iterations or batches of images to generate in total.")

                    st.session_state["defaults"].img2img.batch_size.value = st.number_input("Img2img Batch size", value=st.session_state["defaults"].img2img.batch_size.value,
                                                                                            help="How many images are at once in a batch.\
                                                                                            It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it \
                                                                                            takes to finish generation as more images are generated at once.\
                                                                                            Default: 1")
                    with col4:
                        # Inference Steps
                        st.session_state["defaults"].img2img.num_inference_steps.value = st.number_input("Default Inference Steps",
                                                                                                         value=st.session_state['defaults'].img2img.num_inference_steps.value,
                                                                                                         help="Set the default number of inference steps to use. Default is: 200")

                        st.session_state["defaults"].img2img.num_inference_steps.min_value = st.number_input("Minimum Sampling Steps",
                                                                                                             value=st.session_state['defaults'].img2img.num_inference_steps.min_value,
                                                                                                             help="Set the default minimum value for the inference steps slider. Default is: 10")

                        st.session_state["defaults"].img2img.num_inference_steps.max_value = st.number_input("Maximum Sampling Steps",
                                                                                                             value=st.session_state['defaults'].img2img.num_inference_steps.max_value,
                                                                                                             help="Set the default maximum value for the inference steps slider. Default is: 500")

                        st.session_state["defaults"].img2img.num_inference_steps.step = st.number_input("Inference Slider Steps",
                                                                                                        value=st.session_state['defaults'].img2img.num_inference_steps.step,
                                                                                                        help="Set the default value for the number of steps on the inference steps slider.\
                                                                                                        Default is: 10")

                        # Find Noise Steps
                        st.session_state["defaults"].img2img.find_noise_steps.value = st.number_input("Default Find Noise Steps",
                                                                                                      value=st.session_state['defaults'].img2img.find_noise_steps.value,
                                                                                                      help="Set the default number of find noise steps to use. Default is: 100")

                        st.session_state["defaults"].img2img.find_noise_steps.min_value = st.number_input("Minimum Find Noise Steps",
                                                                                                          value=st.session_state['defaults'].img2img.find_noise_steps.min_value,
                                                                                                          help="Set the default minimum value for the find noise steps slider. Default is: 0")

                        st.session_state["defaults"].img2img.find_noise_steps.step = st.number_input("Find Noise Slider Steps",
                                                                                                     value=st.session_state['defaults'].img2img.find_noise_steps.step,
                                                                                                     help="Set the default value for the number of steps on the find noise steps slider. \
                                                                                                     Default is: 100")

            with col5:
                st.title("General Parameters")

                default_sampler_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", "k_heun", "PLMS", "DDIM"]
                st.session_state["defaults"].img2img.sampler_name = st.selectbox("Default Img2Img Sampler", default_sampler_list,
                                                                                 index=default_sampler_list.index(st.session_state['defaults'].img2img.sampler_name),
                                                                                 help="Defaut sampler to use for img2img. Default: k_euler")

                st.session_state['defaults'].img2img.seed = st.text_input("Default Img2Img Seed", value=st.session_state['defaults'].img2img.seed, help="Default seed.")

                st.session_state["defaults"].img2img.separate_prompts = st.checkbox("Separate Img2Img Prompts", value=st.session_state['defaults'].img2img.separate_prompts,
                                                                                    help="Separate Prompts. Default: False")

                st.session_state["defaults"].img2img.normalize_prompt_weights = st.checkbox("Normalize Img2Img Prompt Weights",
                                                                                            value=st.session_state['defaults'].img2img.normalize_prompt_weights,
                                                                                            help="Choose to normalize prompt weights. Default: True")

                st.session_state["defaults"].img2img.save_individual_images = st.checkbox("Save Individual Img2Img Images",
                                                                                          value=st.session_state['defaults'].img2img.save_individual_images,
                                                                                          help="Choose to save individual images. Default: True")

                st.session_state["defaults"].img2img.save_grid = st.checkbox("Save Img2Img Grid Images",
                                                                             value=st.session_state['defaults'].img2img.save_grid, help="Choose to save the grid images. Default: True")

                st.session_state["defaults"].img2img.group_by_prompt = st.checkbox("Group By Img2Img Prompt",
                                                                                   value=st.session_state['defaults'].img2img.group_by_prompt,
                                                                                   help="Choose to save images grouped by their prompt. Default: False")

                st.session_state["defaults"].img2img.save_as_jpg = st.checkbox("Save Img2Img As JPG", value=st.session_state['defaults'].img2img.save_as_jpg,
                                                                               help="Choose to save images as jpegs. Default: False")

                st.session_state["defaults"].img2img.write_info_files = st.checkbox("Write Info Files For Img2Img Images",
                                                                                    value=st.session_state['defaults'].img2img.write_info_files,
                                                                                    help="Choose to write the info files along with the generated images. Default: True")

                st.session_state["defaults"].img2img.use_GFPGAN = st.checkbox(
                    "Img2Img Use GFPGAN", value=st.session_state['defaults'].img2img.use_GFPGAN, help="Choose to use GFPGAN. Default: False")

                st.session_state["defaults"].img2img.use_RealESRGAN = st.checkbox("Img2Img Use RealESRGAN", value=st.session_state['defaults'].img2img.use_RealESRGAN,
                                                                                  help="Choose to use RealESRGAN. Default: False")

                st.session_state["defaults"].img2img.update_preview = True
                st.session_state["defaults"].img2img.update_preview_frequency = st.number_input("Img2Img Preview Image Update Frequency",
                                                                                                min_value=0,
                                                                                                value=st.session_state['defaults'].img2img.update_preview_frequency,
                                                                                                help="Set the default value for the frrquency of the preview image updates. Default is: 10")

                st.title("Variation Parameters")

                st.session_state["defaults"].img2img.variant_amount = st.number_input("Default Img2Img Variation Amount",
                                                                                      value=st.session_state['defaults'].img2img.variant_amount,
                                                                                      help="Set the default variation to use. Default is: 0.0")

                # I THINK THESE ARE MISSING FROM THE CONFIG FILE
                # st.session_state["defaults"].img2img.variant_amount.min_value = st.number_input("Minimum Img2Img Variation Amount",
                # value=st.session_state['defaults'].img2img.variant_amount.min_value, help="Set the default minimum value for the variation slider. Default is: 0.0"))

                # st.session_state["defaults"].img2img.variant_amount.max_value = st.number_input("Maximum Img2Img Variation Amount",
                # value=st.session_state['defaults'].img2img.variant_amount.max_value, help="Set the default maximum value for the variation slider. Default is: 1.0"))

                # st.session_state["defaults"].img2img.variant_amount.step = st.number_input("Img2Img Variation Slider Steps",
                # value=st.session_state['defaults'].img2img.variant_amount.step, help="Set the default value for the number of steps on the variation slider. Default is: 1"))

                st.session_state['defaults'].img2img.variant_seed = st.text_input("Default Img2Img Variation Seed",
                                                                                  value=st.session_state['defaults'].img2img.variant_seed, help="Default variation seed.")

        with img2txt_tab:
            col1 = st.columns(1, gap="large")

            st.title("Image-To-Text")

            st.session_state["defaults"].img2txt.batch_size = st.number_input("Default Img2Txt Batch Size", value=st.session_state['defaults'].img2txt.batch_size,
                                                                              help="Set the default batch size for Img2Txt. Default is: 420?")

            st.session_state["defaults"].img2txt.blip_image_eval_size = st.number_input("Default Blip Image Size Evaluation",
                                                                                        value=st.session_state['defaults'].img2txt.blip_image_eval_size,
                                                                                        help="Set the default value for the blip image evaluation size. Default is: 512")

        with txt2vid_tab:
            col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

            with col1:
                st.title("Slider Parameters")

                # Width
                st.session_state["defaults"].txt2vid.width.value = st.number_input("Default txt2vid Image Width",
                                                                                   value=st.session_state['defaults'].txt2vid.width.value,
                                                                                   help="Set the default width for the generated image. Default is: 512")

                st.session_state["defaults"].txt2vid.width.min_value = st.number_input("Minimum txt2vid Image Width",
                                                                                       value=st.session_state['defaults'].txt2vid.width.min_value,
                                                                                       help="Set the default minimum value for the width slider. Default is: 64")

                st.session_state["defaults"].txt2vid.width.max_value = st.number_input("Maximum txt2vid Image Width",
                                                                                       value=st.session_state['defaults'].txt2vid.width.max_value,
                                                                                       help="Set the default maximum value for the width slider. Default is: 2048")

                # Height
                st.session_state["defaults"].txt2vid.height.value = st.number_input("Default txt2vid Image Height",
                                                                                    value=st.session_state['defaults'].txt2vid.height.value,
                                                                                    help="Set the default height for the generated image. Default is: 512")

                st.session_state["defaults"].txt2vid.height.min_value = st.number_input("Minimum txt2vid Image Height",
                                                                                        value=st.session_state['defaults'].txt2vid.height.min_value,
                                                                                        help="Set the default minimum value for the height slider. Default is: 64")

                st.session_state["defaults"].txt2vid.height.max_value = st.number_input("Maximum txt2vid Image Height",
                                                                                        value=st.session_state['defaults'].txt2vid.height.max_value,
                                                                                        help="Set the default maximum value for the height slider. Default is: 2048")

                # CFG
                st.session_state["defaults"].txt2vid.cfg_scale.value = st.number_input("Default txt2vid CFG Scale",
                                                                                       value=st.session_state['defaults'].txt2vid.cfg_scale.value,
                                                                                       help="Set the default value for the CFG Scale. Default is: 7.5")

                st.session_state["defaults"].txt2vid.cfg_scale.min_value = st.number_input("Minimum txt2vid CFG Scale Value",
                                                                                           value=st.session_state['defaults'].txt2vid.cfg_scale.min_value,
                                                                                           help="Set the default minimum value for the CFG scale slider. Default is: 1")

                st.session_state["defaults"].txt2vid.cfg_scale.step = st.number_input("txt2vid CFG Slider Steps",
                                                                                      value=st.session_state['defaults'].txt2vid.cfg_scale.step,
                                                                                      help="Set the default value for the number of steps on the CFG scale slider. Default is: 0.5")

                with col2:
                    # Sampling Steps
                    st.session_state["defaults"].txt2vid.sampling_steps.value = st.number_input("Default txt2vid Sampling Steps",
                                                                                                value=st.session_state['defaults'].txt2vid.sampling_steps.value,
                                                                                                help="Set the default number of sampling steps to use. Default is: 30 (with k_euler)")

                    st.session_state["defaults"].txt2vid.sampling_steps.min_value = st.number_input("Minimum txt2vid Sampling Steps",
                                                                                                    value=st.session_state['defaults'].txt2vid.sampling_steps.min_value,
                                                                                                    help="Set the default minimum value for the sampling steps slider. Default is: 1")

                    st.session_state["defaults"].txt2vid.sampling_steps.step = st.number_input("txt2vid Sampling Slider Steps",
                                                                                               value=st.session_state['defaults'].txt2vid.sampling_steps.step,
                                                                                               help="Set the default value for the number of steps on the sampling steps slider. Default is: 10")

                    # Batch Count
                    st.session_state["defaults"].txt2vid.batch_count.value = st.number_input("txt2vid Batch count", value=st.session_state['defaults'].txt2vid.batch_count.value,
                                                                                             help="How many iterations or batches of images to generate in total.")

                    st.session_state["defaults"].txt2vid.batch_size.value = st.number_input("txt2vid Batch size", value=st.session_state.defaults.txt2vid.batch_size.value,
                                                                                            help="How many images are at once in a batch.\
                                                                                            It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it \
                                                                                            takes to finish generation as more images are generated at once.\
                                                                                            Default: 1")

                    # Inference Steps
                    st.session_state["defaults"].txt2vid.num_inference_steps.value = st.number_input("Default Txt2Vid Inference Steps",
                                                                                                     value=st.session_state['defaults'].txt2vid.num_inference_steps.value,
                                                                                                     help="Set the default number of inference steps to use. Default is: 200")

                    st.session_state["defaults"].txt2vid.num_inference_steps.min_value = st.number_input("Minimum Txt2Vid Sampling Steps",
                                                                                                         value=st.session_state['defaults'].txt2vid.num_inference_steps.min_value,
                                                                                                         help="Set the default minimum value for the inference steps slider. Default is: 10")

                    st.session_state["defaults"].txt2vid.num_inference_steps.max_value = st.number_input("Maximum Txt2Vid Sampling Steps",
                                                                                                         value=st.session_state['defaults'].txt2vid.num_inference_steps.max_value,
                                                                                                         help="Set the default maximum value for the inference steps slider. Default is: 500")
                    st.session_state["defaults"].txt2vid.num_inference_steps.step = st.number_input("Txt2Vid Inference Slider Steps",
                                                                                                    value=st.session_state['defaults'].txt2vid.num_inference_steps.step,
                                                                                                    help="Set the default value for the number of steps on the inference steps slider. Default is: 10")

            with col3:
                st.title("General Parameters")

                st.session_state['defaults'].txt2vid.default_model = st.text_input("Default Txt2Vid Model", value=st.session_state['defaults'].txt2vid.default_model,
                                                                                   help="Default: CompVis/stable-diffusion-v1-4")

                # INSERT CUSTOM_MODELS_LIST HERE

                default_sampler_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", "k_heun", "PLMS", "DDIM"]
                st.session_state["defaults"].txt2vid.default_sampler = st.selectbox("Default txt2vid Sampler", default_sampler_list,
                                                                                    index=default_sampler_list.index(st.session_state['defaults'].txt2vid.default_sampler),
                                                                                    help="Defaut sampler to use for txt2vid. Default: k_euler")

                st.session_state['defaults'].txt2vid.seed = st.text_input("Default txt2vid Seed", value=st.session_state['defaults'].txt2vid.seed, help="Default seed.")

                st.session_state['defaults'].txt2vid.scheduler_name = st.text_input("Default Txt2Vid Scheduler",
                                                                                    value=st.session_state['defaults'].txt2vid.scheduler_name, help="Default scheduler.")

                st.session_state["defaults"].txt2vid.separate_prompts = st.checkbox("Separate txt2vid Prompts",
                                                                                    value=st.session_state['defaults'].txt2vid.separate_prompts, help="Separate Prompts. Default: False")

                st.session_state["defaults"].txt2vid.normalize_prompt_weights = st.checkbox("Normalize txt2vid Prompt Weights",
                                                                                            value=st.session_state['defaults'].txt2vid.normalize_prompt_weights,
                                                                                            help="Choose to normalize prompt weights. Default: True")

                st.session_state["defaults"].txt2vid.save_individual_images = st.checkbox("Save Individual txt2vid Images",
                                                                                          value=st.session_state['defaults'].txt2vid.save_individual_images,
                                                                                          help="Choose to save individual images. Default: True")

                st.session_state["defaults"].txt2vid.save_video = st.checkbox("Save Txt2Vid Video", value=st.session_state['defaults'].txt2vid.save_video,
                                                                              help="Choose to save the Txt2Vid video. Default: True")

                st.session_state["defaults"].txt2vid.save_video_on_stop = st.checkbox("Save video on Stop", value=st.session_state['defaults'].txt2vid.save_video_on_stop,
                                                                                      help="Save a video with all the images generated as frames when we hit the stop button \
																					  during a generation.")

                st.session_state["defaults"].txt2vid.group_by_prompt = st.checkbox("Group By txt2vid Prompt", value=st.session_state['defaults'].txt2vid.group_by_prompt,
                                                                                   help="Choose to save images grouped by their prompt. Default: False")

                st.session_state["defaults"].txt2vid.save_as_jpg = st.checkbox("Save txt2vid As JPG", value=st.session_state['defaults'].txt2vid.save_as_jpg,
                                                                               help="Choose to save images as jpegs. Default: False")

                # Need more info for the Help dialog...
                st.session_state["defaults"].txt2vid.do_loop = st.checkbox("Loop Generations", value=st.session_state['defaults'].txt2vid.do_loop,
                                                                           help="Choose to loop or something, IDK.... Default: False")

                st.session_state["defaults"].txt2vid.max_duration_in_seconds = st.number_input("Txt2Vid Max Duration in Seconds", value=st.session_state['defaults'].txt2vid.max_duration_in_seconds,
                                                                                  help="Set the default value for the max duration in seconds for the video generated. Default is: 30")

                st.session_state["defaults"].txt2vid.write_info_files = st.checkbox("Write Info Files For txt2vid Images", value=st.session_state['defaults'].txt2vid.write_info_files,
                                                                                    help="Choose to write the info files along with the generated images. Default: True")

                st.session_state["defaults"].txt2vid.use_GFPGAN = st.checkbox("txt2vid Use GFPGAN", value=st.session_state['defaults'].txt2vid.use_GFPGAN,
                                                                              help="Choose to use GFPGAN. Default: False")

                st.session_state["defaults"].txt2vid.use_RealESRGAN = st.checkbox("txt2vid Use RealESRGAN", value=st.session_state['defaults'].txt2vid.use_RealESRGAN,
                                                                                  help="Choose to use RealESRGAN. Default: False")

                st.session_state["defaults"].txt2vid.update_preview = True
                st.session_state["defaults"].txt2vid.update_preview_frequency = st.number_input("txt2vid Preview Image Update Frequency",
                                                                                                value=st.session_state['defaults'].txt2vid.update_preview_frequency,
                                                                                                help="Set the default value for the frrquency of the preview image updates. Default is: 10")

            with col4:
                st.title("Variation Parameters")

                st.session_state["defaults"].txt2vid.variant_amount.value = st.number_input("Default txt2vid Variation Amount",
                                                                                            value=st.session_state['defaults'].txt2vid.variant_amount.value,
                                                                                            help="Set the default variation to use. Default is: 0.0")

                st.session_state["defaults"].txt2vid.variant_amount.min_value = st.number_input("Minimum txt2vid Variation Amount",
                                                                                                value=st.session_state['defaults'].txt2vid.variant_amount.min_value,
                                                                                                help="Set the default minimum value for the variation slider. Default is: 0.0")

                st.session_state["defaults"].txt2vid.variant_amount.max_value = st.number_input("Maximum txt2vid Variation Amount",
                                                                                                value=st.session_state['defaults'].txt2vid.variant_amount.max_value,
                                                                                                help="Set the default maximum value for the variation slider. Default is: 1.0")

                st.session_state["defaults"].txt2vid.variant_amount.step = st.number_input("txt2vid Variation Slider Steps",
                                                                                           value=st.session_state['defaults'].txt2vid.variant_amount.step,
                                                                                           help="Set the default value for the number of steps on the variation slider. Default is: 1")

                st.session_state['defaults'].txt2vid.variant_seed = st.text_input("Default txt2vid Variation Seed",
                                                                                  value=st.session_state['defaults'].txt2vid.variant_seed, help="Default variation seed.")

            with col5:
                st.title("Beta Parameters")

                # Beta Start
                st.session_state["defaults"].txt2vid.beta_start.value = st.number_input("Default txt2vid Beta Start Value",
                                                                                        value=st.session_state['defaults'].txt2vid.beta_start.value,
                                                                                        help="Set the default variation to use. Default is: 0.0")

                st.session_state["defaults"].txt2vid.beta_start.min_value = st.number_input("Minimum txt2vid Beta Start Amount",
                                                                                            value=st.session_state['defaults'].txt2vid.beta_start.min_value,
                                                                                            help="Set the default minimum value for the variation slider. Default is: 0.0")

                st.session_state["defaults"].txt2vid.beta_start.max_value = st.number_input("Maximum txt2vid Beta Start Amount",
                                                                                            value=st.session_state['defaults'].txt2vid.beta_start.max_value,
                                                                                            help="Set the default maximum value for the variation slider. Default is: 1.0")

                st.session_state["defaults"].txt2vid.beta_start.step = st.number_input("txt2vid Beta Start Slider Steps", value=st.session_state['defaults'].txt2vid.beta_start.step,
                                                                                       help="Set the default value for the number of steps on the variation slider. Default is: 1")

                st.session_state["defaults"].txt2vid.beta_start.format = st.text_input("Default txt2vid Beta Start Format", value=st.session_state['defaults'].txt2vid.beta_start.format,
                                                                                       help="Set the default Beta Start Format. Default is: %.5\f")

                # Beta End
                st.session_state["defaults"].txt2vid.beta_end.value = st.number_input("Default txt2vid Beta End Value", value=st.session_state['defaults'].txt2vid.beta_end.value,
                                                                                      help="Set the default variation to use. Default is: 0.0")

                st.session_state["defaults"].txt2vid.beta_end.min_value = st.number_input("Minimum txt2vid Beta End Amount", value=st.session_state['defaults'].txt2vid.beta_end.min_value,
                                                                                          help="Set the default minimum value for the variation slider. Default is: 0.0")

                st.session_state["defaults"].txt2vid.beta_end.max_value = st.number_input("Maximum txt2vid Beta End Amount", value=st.session_state['defaults'].txt2vid.beta_end.max_value,
                                                                                          help="Set the default maximum value for the variation slider. Default is: 1.0")

                st.session_state["defaults"].txt2vid.beta_end.step = st.number_input("txt2vid Beta End Slider Steps", value=st.session_state['defaults'].txt2vid.beta_end.step,
                                                                                     help="Set the default value for the number of steps on the variation slider. Default is: 1")

                st.session_state["defaults"].txt2vid.beta_end.format = st.text_input("Default txt2vid Beta End Format", value=st.session_state['defaults'].txt2vid.beta_start.format,
                                                                                     help="Set the default Beta Start Format. Default is: %.5\f")

        with image_processing:
            col1, col2, col3, col4, col5 = st.columns(5, gap="large")

            with col1:
                st.title("GFPGAN")

                st.session_state["defaults"].gfpgan.strength = st.number_input("Default Img2Txt Batch Size", value=st.session_state['defaults'].gfpgan.strength,
                                                                               help="Set the default global strength for GFPGAN. Default is: 100")
            with col2:
                st.title("GoBig")
            with col3:
                st.title("RealESRGAN")
            with col4:
                st.title("LDSR")
            with col5:
                st.title("GoLatent")

        with textual_inversion_tab:
            st.title("Textual Inversion")

            st.session_state['defaults'].textual_inversion.pretrained_model_name_or_path = st.text_input("Default Textual Inversion Model Path",
                                                                                                         value=st.session_state['defaults'].textual_inversion.pretrained_model_name_or_path,
                                                                                                         help="Default: models/ldm/stable-diffusion-v1-4")

            st.session_state['defaults'].textual_inversion.tokenizer_name = st.text_input("Default Img2Img Variation Seed", value=st.session_state['defaults'].textual_inversion.tokenizer_name,
                                                                                          help="Default tokenizer seed.")

        with concepts_library_tab:
            st.title("Concepts Library")
            #st.info("Under Construction. :construction_worker:")
            col1, col2, col3, col4, col5 = st.columns(5, gap='large')
            with col1:
                st.session_state["defaults"].concepts_library.concepts_per_page = st.number_input("Concepts Per Page", value=st.session_state['defaults'].concepts_library.concepts_per_page,
                                                                                                  help="Number of concepts per page to show on the Concepts Library. Default: '12'")

        # add space for the buttons at the bottom
        st.markdown("---")

        # We need a submit button to save the Settings
        # as well as one to reset them to the defaults, just in case.
        _, _, save_button_col, reset_button_col, _, _ = st.columns([1, 1, 1, 1, 1, 1], gap="large")
        with save_button_col:
            save_button = st.form_submit_button("Save")

        with reset_button_col:
            reset_button = st.form_submit_button("Reset")

        if save_button:
            OmegaConf.save(config=st.session_state.defaults, f="configs/webui/userconfig_streamlit.yaml")
            loaded = OmegaConf.load("configs/webui/userconfig_streamlit.yaml")
            assert st.session_state.defaults == loaded

            #
            if (os.path.exists(".streamlit/config.toml")):
                with open(".streamlit/config.toml", "w") as toml_file:
                    toml.dump(st.session_state["streamlit_config"], toml_file)

        if reset_button:
            st.session_state["defaults"] = OmegaConf.load("configs/webui/webui_streamlit.yaml")
            st.experimental_rerun()

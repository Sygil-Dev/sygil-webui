# This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

# Copyright 2022 sd-webui team.
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
from streamlit import StopException
#from streamlit.elements import image as STImage
import streamlit.components.v1 as components
from streamlit.runtime.media_file_manager  import media_file_manager
from streamlit.elements.image import image_to_url

#other imports
import uuid
from typing import Union
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# Temp imports


# end of imports
#---------------------------------------------------------------------------------------------------------------


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

#
# Dev mode (server)
# _component_func = components.declare_component(
#         "sd-gallery",
#         url="http://localhost:3001",
#     )

# Init Vuejs component
_component_func = components.declare_component(
    "sd-gallery", "./frontend/dists/sd-gallery/dist")

def sdGallery(images=[], key=None):
    component_value = _component_func(images=imgsToGallery(images), key=key, default="")
    return component_value

def imgsToGallery(images):
    urls = []
    for i in images:
        # random string for id
        random_id = str(uuid.uuid4())
        url = image_to_url(
            image=i,
            image_id= random_id,
            width=i.width,
            clamp=False,
            channels="RGB",
            output_format="PNG"
        )
        # image_io = BytesIO()
        # i.save(image_io, 'PNG')
        # width, height = i.size
        # image_id = "%s" % (str(images.index(i)))
        # (data, mimetype) = STImage._normalize_to_bytes(image_io.getvalue(), width, 'auto')
        # this_file = media_file_manager.add(data, mimetype, image_id)
        # img_str = this_file.url
        urls.append(url)

    return urls


class plugin_info():
    plugname = "txt2img"
    description = "Text to Image"
    isTab = True
    displayPriority = 1

#
def txt2img(prompt: str, ddim_steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, seed: Union[int, str, None],
            height: int, width: int, separate_prompts:bool = False, normalize_prompt_weights:bool = True,
            save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
            save_as_jpg: bool = True, use_GFPGAN: bool = True, GFPGAN_model: str = 'GFPGANv1.3', use_RealESRGAN: bool = False,
            RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B", use_LDSR: bool = True, LDSR_model: str = "model",
            fp = None, variant_amount: float = None,
            variant_seed: int = None, ddim_eta:float = 0.0, write_info_files:bool = True):

    outpath = st.session_state['defaults'].general.outdir_txt2img

    seed = seed_to_int(seed)

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(server_state["model"])
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(server_state["model"])
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(server_state["model"],'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(server_state["model"],'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(server_state["model"],'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(server_state["model"],'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(server_state["model"],'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(server_state["model"],'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale,
                                         unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=generation_callback,
                                                 log_every_t=int(st.session_state.update_preview_frequency))

        return samples_ddim

    #try:
    output_images, seed, info, stats = process_images(
        outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_name=sampler_name,
                save_grid=save_grid,
                batch_size=batch_size,
                n_iter=n_iter,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=separate_prompts,
                use_GFPGAN=st.session_state["use_GFPGAN"],
                GFPGAN_model=st.session_state["GFPGAN_model"],
                use_RealESRGAN=st.session_state["use_RealESRGAN"],
                realesrgan_model_name=RealESRGAN_model,
                use_LDSR=st.session_state["use_LDSR"],
                LDSR_model_name=LDSR_model,
                ddim_eta=ddim_eta,
                normalize_prompt_weights=normalize_prompt_weights,
                save_individual_images=save_individual_images,
                sort_samples=group_by_prompt,
                write_info_files=write_info_files,
                jpg_sample=save_as_jpg,
                variant_amount=variant_amount,
                variant_seed=variant_seed,
    )

    del sampler

    return output_images, seed, info, stats

    #except RuntimeError as e:
        #err = e
        #err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        #stats = err_msg
        #return [], seed, 'err', stats

#
def layout():
    with st.form("txt2img-inputs"):
        st.session_state["generation_mode"] = "txt2img"

        input_col1, generate_col1 = st.columns([10,1])

        with input_col1:
            #prompt = st.text_area("Input Text","")
            prompt = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")

        # creating the page layout using columns
        col1, col2, col3 = st.columns([1,2,1], gap="large")

        with col1:
            width = st.slider("Width:", min_value=st.session_state['defaults'].txt2img.width.min_value, max_value=st.session_state['defaults'].txt2img.width.max_value,
                              value=st.session_state['defaults'].txt2img.width.value, step=st.session_state['defaults'].txt2img.width.step)
            height = st.slider("Height:", min_value=st.session_state['defaults'].txt2img.height.min_value, max_value=st.session_state['defaults'].txt2img.height.max_value,
                               value=st.session_state['defaults'].txt2img.height.value, step=st.session_state['defaults'].txt2img.height.step)
            cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=st.session_state['defaults'].txt2img.cfg_scale.min_value,
                                  max_value=st.session_state['defaults'].txt2img.cfg_scale.max_value,
                                  value=st.session_state['defaults'].txt2img.cfg_scale.value, step=st.session_state['defaults'].txt2img.cfg_scale.step,
                                  help="How strongly the image should follow the prompt.")
            seed = st.text_input("Seed:", value=st.session_state['defaults'].txt2img.seed, help=" The seed to use, if left blank a random seed will be generated.")

            with st.expander("Batch Options"):
                #batch_count = st.slider("Batch count.", min_value=st.session_state['defaults'].txt2img.batch_count.min_value, max_value=st.session_state['defaults'].txt2img.batch_count.max_value,
                                        #value=st.session_state['defaults'].txt2img.batch_count.value, step=st.session_state['defaults'].txt2img.batch_count.step,
                                        #help="How many iterations or batches of images to generate in total.")

                #batch_size = st.slider("Batch size", min_value=st.session_state['defaults'].txt2img.batch_size.min_value, max_value=st.session_state['defaults'].txt2img.batch_size.max_value,
                                       #value=st.session_state.defaults.txt2img.batch_size.value, step=st.session_state.defaults.txt2img.batch_size.step,
                                       #help="How many images are at once in a batch.\
                                       #It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
                                       #Default: 1")

                st.session_state["batch_count"] = int(st.text_input("Batch count.", value=st.session_state['defaults'].txt2img.batch_count.value,
                                                                help="How many iterations or batches of images to generate in total."))

                st.session_state["batch_size"] = int(st.text_input("Batch size", value=st.session_state.defaults.txt2img.batch_size.value,
                                                                   help="How many images are at once in a batch.\
                                                                   It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes \
                                                                   to finish generation as more images are generated at once.\
                                                                   Default: 1") )

            with st.expander("Preview Settings"):

                st.session_state["update_preview"] = st.session_state["defaults"].general.update_preview
                st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=st.session_state['defaults'].txt2img.update_preview_frequency,
                                                                             help="Frequency in steps at which the the preview image is updated. By default the frequency \
                                                                              is set to 10 step.")

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


                st.session_state["progress_bar_text"] = st.empty()
                st.session_state["progress_bar_text"].info("Nothing but crickets here, try generating something first.")

                st.session_state["progress_bar"] = st.empty()

                message = st.empty()

            with gallery_tab:
                st.session_state["gallery"] = st.empty()
                st.session_state["gallery"].info("Nothing but crickets here, try generating something first.")

        with col3:
            # If we have custom models available on the "models/custom"
            #folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
            custom_models_available()

            if server_state["CustomModel_available"]:
                st.session_state["custom_model"] = st.selectbox("Custom Model:", server_state["custom_models"],
                                                                index=server_state["custom_models"].index(st.session_state['defaults'].general.default_model),
                                                                help="Select the model you want to use. This option is only available if you have custom models \
                                                                on your 'models/custom' folder. The model name that will be shown here is the same as the name\
                                                                the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
                                                                will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")

            st.session_state.sampling_steps = st.number_input("Sampling Steps", value=st.session_state.defaults.txt2img.sampling_steps.value,
                                                              min_value=st.session_state.defaults.txt2img.sampling_steps.min_value,
                                                              step=st.session_state['defaults'].txt2img.sampling_steps.step,
                                                              help="Set the default number of sampling steps to use. Default is: 30 (with k_euler)")

            sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
            sampler_name = st.selectbox("Sampling method", sampler_name_list,
                                        index=sampler_name_list.index(st.session_state['defaults'].txt2img.default_sampler), help="Sampling method to use. Default: k_euler")

            with st.expander("Advanced"):
                with st.expander("Output Settings"):
                    separate_prompts = st.checkbox("Create Prompt Matrix.", value=st.session_state['defaults'].txt2img.separate_prompts,
                                                   help="Separate multiple prompts using the `|` character, and get all combinations of them.")

                    normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=st.session_state['defaults'].txt2img.normalize_prompt_weights,
                                                           help="Ensure the sum of all weights add up to 1.0")

                    save_individual_images = st.checkbox("Save individual images.", value=st.session_state['defaults'].txt2img.save_individual_images,
                                                         help="Save each image generated before any filter or enhancement is applied.")

                    save_grid = st.checkbox("Save grid",value=st.session_state['defaults'].txt2img.save_grid, help="Save a grid with all the images generated into a single image.")
                    group_by_prompt = st.checkbox("Group results by prompt", value=st.session_state['defaults'].txt2img.group_by_prompt,
                                                  help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")

                    write_info_files = st.checkbox("Write Info file", value=st.session_state['defaults'].txt2img.write_info_files,
                                                   help="Save a file next to the image with informartion about the generation.")

                    save_as_jpg = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].txt2img.save_as_jpg, help="Saves the images as jpg instead of png.")

                # check if GFPGAN, RealESRGAN and LDSR are available.
                #if "GFPGAN_available" not in st.session_state:
                GFPGAN_available()

                #if "RealESRGAN_available" not in st.session_state:
                RealESRGAN_available()

                #if "LDSR_available" not in st.session_state:
                LDSR_available()

                if st.session_state["GFPGAN_available"] or st.session_state["RealESRGAN_available"] or st.session_state["LDSR_available"]:
                    with st.expander("Post-Processing"):
                        face_restoration_tab, upscaling_tab = st.tabs(["Face Restoration", "Upscaling"])
                        with face_restoration_tab:
                            # GFPGAN used for face restoration
                            if st.session_state["GFPGAN_available"]:
                                #with st.expander("Face Restoration"):
                                #if st.session_state["GFPGAN_available"]:
                                #with st.expander("GFPGAN"):
                                    st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].txt2img.use_GFPGAN,
                                                                                 help="Uses the GFPGAN model to improve faces after the generation.\
                                                                                 This greatly improve the quality and consistency of faces but uses\
                                                                                 extra VRAM. Disable if you need the extra VRAM.")

                                    st.session_state["GFPGAN_model"] = st.selectbox("GFPGAN model", st.session_state["GFPGAN_models"],
                                                                                    index=st.session_state["GFPGAN_models"].index(st.session_state['defaults'].general.GFPGAN_model))

                                #st.session_state["GFPGAN_strenght"] = st.slider("Effect Strenght", min_value=1, max_value=100, value=1, step=1, help='')

                            else:
                                st.session_state["use_GFPGAN"] = False

                        with upscaling_tab:
                            st.session_state['use_upscaling'] = st.checkbox("Use Upscaling", value=st.session_state['defaults'].txt2img.use_upscaling)

                            # RealESRGAN and LDSR used for upscaling.
                            if st.session_state["RealESRGAN_available"] or st.session_state["LDSR_available"]:

                                upscaling_method_list = []
                                if st.session_state["RealESRGAN_available"]:
                                    upscaling_method_list.append("RealESRGAN")
                                if st.session_state["LDSR_available"]:
                                    upscaling_method_list.append("LDSR")

                                #print (st.session_state["RealESRGAN_available"])
                                st.session_state["upscaling_method"] = st.selectbox("Upscaling Method", upscaling_method_list,
                                                                                    index=upscaling_method_list.index(str(st.session_state['defaults'].general.upscaling_method)))

                                if st.session_state["RealESRGAN_available"]:
                                    with st.expander("RealESRGAN"):
                                        if st.session_state["upscaling_method"] == "RealESRGAN" and st.session_state['use_upscaling']:
                                            st.session_state["use_RealESRGAN"] = True
                                        else:
                                            st.session_state["use_RealESRGAN"] = False

                                        st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", st.session_state["RealESRGAN_models"],
                                                                                        index=st.session_state["RealESRGAN_models"].index(st.session_state['defaults'].general.RealESRGAN_model))
                                else:
                                    st.session_state["use_RealESRGAN"] = False
                                    st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"


                                #
                                if st.session_state["LDSR_available"]:
                                    with st.expander("LDSR"):
                                        if st.session_state["upscaling_method"] == "LDSR" and st.session_state['use_upscaling']:
                                            st.session_state["use_LDSR"] = True
                                        else:
                                            st.session_state["use_LDSR"] = False

                                        st.session_state["LDSR_model"] = st.selectbox("LDSR model", st.session_state["LDSR_models"],
                                                                                      index=st.session_state["LDSR_models"].index(st.session_state['defaults'].general.LDSR_model))

                                        st.session_state["ldsr_sampling_steps"] = int(st.text_input("Sampling Steps", value=st.session_state['defaults'].txt2img.LDSR_config.sampling_steps,
                                                                                      help=""))

                                        st.session_state["preDownScale"] = int(st.text_input("PreDownScale", value=st.session_state['defaults'].txt2img.LDSR_config.preDownScale,
                                                                               help=""))

                                        st.session_state["postDownScale"] = int(st.text_input("postDownScale", value=st.session_state['defaults'].txt2img.LDSR_config.postDownScale,
                                                                               help=""))

                                        downsample_method_list = ['Nearest', 'Lanczos']
                                        st.session_state["downsample_method"] = st.selectbox("Downsample Method", downsample_method_list,
                                                                                             index=downsample_method_list.index(st.session_state['defaults'].txt2img.LDSR_config.downsample_method))

                                else:
                                    st.session_state["use_LDSR"] = False
                                    st.session_state["LDSR_model"] = "model"

                with st.expander("Variant"):
                    variant_amount = st.slider("Variant Amount:", value=st.session_state['defaults'].txt2img.variant_amount.value,
                                               min_value=st.session_state['defaults'].txt2img.variant_amount.min_value, max_value=st.session_state['defaults'].txt2img.variant_amount.max_value,
                                               step=st.session_state['defaults'].txt2img.variant_amount.step)
                    variant_seed = st.text_input("Variant Seed:", value=st.session_state['defaults'].txt2img.seed,
                                                 help="The seed to use when generating a variant, if left blank a random seed will be generated.")

            #galleryCont = st.empty()

        # Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
        generate_col1.write("")
        generate_col1.write("")
        generate_button = generate_col1.form_submit_button("Generate")

        #
        if generate_button:

            with col2:
                with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders,index=[0]):
                    load_models(use_LDSR=st.session_state["use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
                                use_GFPGAN=st.session_state["use_GFPGAN"], GFPGAN_model=st.session_state["GFPGAN_model"] ,
                                use_RealESRGAN=st.session_state["use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"],
                                CustomModel_available=server_state["CustomModel_available"], custom_model=st.session_state["custom_model"])


                #print(st.session_state['use_RealESRGAN'])
                #print(st.session_state['use_LDSR'])
                #try:
                #

                output_images, seeds, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, st.session_state["batch_count"], st.session_state["batch_size"],
                                                            cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
                                                            save_grid, group_by_prompt, save_as_jpg, st.session_state["use_GFPGAN"], st.session_state['GFPGAN_model'],
                                                            use_RealESRGAN=st.session_state["use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"],
                                                            use_LDSR=st.session_state["use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
                                                            variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files)

                message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")

            #history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont = st.session_state['historyTab']

            #if 'latestImages' in st.session_state:
                #for i in output_images:
                    ##push the new image to the list of latest images and remove the oldest one
                    ##remove the last index from the list\
                    #st.session_state['latestImages'].pop()
                    ##add the new image to the start of the list
                    #st.session_state['latestImages'].insert(0, i)
                #PlaceHolder.empty()
                #with PlaceHolder.container():
                    #col1, col2, col3 = st.columns(3)
                    #col1_cont = st.container()
                    #col2_cont = st.container()
                    #col3_cont = st.container()
                    #images = st.session_state['latestImages']
                    #with col1_cont:
                        #with col1:
                            #[st.image(images[index]) for index in [0, 3, 6] if index < len(images)]
                    #with col2_cont:
                        #with col2:
                            #[st.image(images[index]) for index in [1, 4, 7] if index < len(images)]
                    #with col3_cont:
                        #with col3:
                            #[st.image(images[index]) for index in [2, 5, 8] if index < len(images)]
                    #historyGallery = st.empty()

                ## check if output_images length is the same as seeds length
                #with gallery_tab:
                    #st.markdown(createHTMLGallery(output_images,seeds), unsafe_allow_html=True)


                    #st.session_state['historyTab'] = [history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont]

            with gallery_tab:
                print(seeds)
                sdGallery(output_images)


            #except (StopException, KeyError):
                #print(f"Received Streamlit StopException")

                # this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
                # use the current col2 first tab to show the preview_img and update it as its generated.
                #preview_image.image(output_images)



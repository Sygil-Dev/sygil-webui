# This file is part of sygil-webui (https://github.com/Sygil-Dev/sandbox-webui/).

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
#from sd_utils import *
from sd_utils import st, server_state, torch_gc, \
     RealESRGAN_available, GFPGAN_available, \
     LDSR_available, load_models, logger, load_GFPGAN, load_RealESRGAN, \
     load_LDSR

# streamlit imports

#streamlit components section
import hydralit_components as hc

#other imports
import os
from PIL import Image
import torch

# Temp imports

# end of imports
#---------------------------------------------------------------------------------------------------------------

def post_process(use_GFPGAN=True, GFPGAN_model='', use_RealESRGAN=False, realesrgan_model_name="", use_LDSR=False, LDSR_model_name=""):

    for i in range(len(st.session_state["uploaded_image"])):
        #st.session_state["uploaded_image"][i].pil_image

        if use_GFPGAN and server_state["GFPGAN"] is not None and not use_RealESRGAN and not use_LDSR:
            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].text("Running GFPGAN on image %d of %d..." % (i+1, len(st.session_state["uploaded_image"])))

            if "progress_bar" in st.session_state:
                st.session_state["progress_bar"].progress(
                    int(100 * float(i+1 if i+1 < len(st.session_state["uploaded_image"]) else len(st.session_state["uploaded_image"]))/float(len(st.session_state["uploaded_image"]))))

            if server_state["GFPGAN"].name != GFPGAN_model:
                load_models(use_LDSR=use_LDSR, LDSR_model=LDSR_model_name, use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

            torch_gc()

            with torch.autocast('cuda'):
                cropped_faces, restored_faces, restored_img = server_state["GFPGAN"].enhance(st.session_state["uploaded_image"][i].pil_image, has_aligned=False, only_center_face=False, paste_back=True)

            gfpgan_sample = restored_img[:,:,::-1]
            gfpgan_image = Image.fromarray(gfpgan_sample)

            #if st.session_state["GFPGAN_strenght"]:
                #gfpgan_sample = Image.blend(image, gfpgan_image, st.session_state["GFPGAN_strenght"])

            gfpgan_filename = st.session_state["uploaded_image"][i].name.split('.')[0] + '-gfpgan'

            gfpgan_image.save(os.path.join(st.session_state["defaults"].post_processing.outdir_post_processing, f"{gfpgan_filename}.png"))

        #
        elif use_RealESRGAN and server_state["RealESRGAN"] is not None and not use_GFPGAN:
            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].text("Running RealESRGAN on image %d of %d..." % (i+1, len(st.session_state["uploaded_image"])))

            if "progress_bar" in st.session_state:
                st.session_state["progress_bar"].progress(
                    int(100 * float(i+1 if i+1 < len(st.session_state["uploaded_image"]) else len(st.session_state["uploaded_image"]))/float(len(st.session_state["uploaded_image"]))))

            torch_gc()

            if server_state["RealESRGAN"].model.name != realesrgan_model_name:
                #try_loading_RealESRGAN(realesrgan_model_name)
                load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

            output, img_mode = server_state["RealESRGAN"].enhance(st.session_state["uploaded_image"][i].pil_image)
            esrgan_filename = st.session_state["uploaded_image"][i].name.split('.')[0] + '-esrgan4x'
            esrgan_sample = output[:,:,::-1]
            esrgan_image = Image.fromarray(esrgan_sample)

            esrgan_image.save(os.path.join(st.session_state["defaults"].post_processing.outdir_post_processing, f"{esrgan_filename}.png"))

        #
        elif use_LDSR and "LDSR" in server_state and not use_GFPGAN:
            logger.info ("Running LDSR on image %d of %d..." % (i+1, len(st.session_state["uploaded_image"])))
            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].text("Running LDSR on image %d of %d..." % (i+1, len(st.session_state["uploaded_image"])))
            if "progress_bar" in st.session_state:
                st.session_state["progress_bar"].progress(
                    int(100 * float(i+1 if i+1 < len(st.session_state["uploaded_image"]) else len(st.session_state["uploaded_image"]))/float(len(st.session_state["uploaded_image"]))))

            torch_gc()

            if server_state["LDSR"].name != LDSR_model_name:
                #try_loading_RealESRGAN(realesrgan_model_name)
                load_models(use_LDSR=use_LDSR, LDSR_model=LDSR_model_name, use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

            result = server_state["LDSR"].superResolution(st.session_state["uploaded_image"][i].pil_image, ddimSteps = st.session_state["ldsr_sampling_steps"],
                                                          preDownScale = st.session_state["preDownScale"], postDownScale = st.session_state["postDownScale"],
                                                          downsample_method=st.session_state["downsample_method"])

            ldsr_filename = st.session_state["uploaded_image"][i].name.split('.')[0] + '-ldsr4x'

            result.save(os.path.join(st.session_state["defaults"].post_processing.outdir_post_processing, f"{ldsr_filename}.png"))

        #
        elif use_LDSR and "LDSR" in server_state and use_GFPGAN and "GFPGAN" in server_state:
            logger.info ("Running GFPGAN+LDSR on image %d of %d..." % (i+1, len(st.session_state["uploaded_image"])))
            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].text("Running GFPGAN+LDSR on image %d of %d..." % (i+1, len(st.session_state["uploaded_image"])))

            if "progress_bar" in st.session_state:
                st.session_state["progress_bar"].progress(
                    int(100 * float(i+1 if i+1 < len(st.session_state["uploaded_image"]) else len(st.session_state["uploaded_image"]))/float(len(st.session_state["uploaded_image"]))))

            if server_state["GFPGAN"].name != GFPGAN_model:
                load_models(use_LDSR=use_LDSR, LDSR_model=LDSR_model_name, use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

            torch_gc()
            cropped_faces, restored_faces, restored_img = server_state["GFPGAN"].enhance(st.session_state["uploaded_image"][i].pil_image, has_aligned=False, only_center_face=False, paste_back=True)

            gfpgan_sample = restored_img[:,:,::-1]
            gfpgan_image = Image.fromarray(gfpgan_sample)

            if server_state["LDSR"].name != LDSR_model_name:
                #try_loading_RealESRGAN(realesrgan_model_name)
                load_models(use_LDSR=use_LDSR, LDSR_model=LDSR_model_name, use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

            #LDSR.superResolution(gfpgan_image, ddimSteps=100, preDownScale='None', postDownScale='None', downsample_method="Lanczos")
            result = server_state["LDSR"].superResolution(gfpgan_image, ddimSteps = st.session_state["ldsr_sampling_steps"],
                                                          preDownScale = st.session_state["preDownScale"], postDownScale = st.session_state["postDownScale"],
                                                          downsample_method=st.session_state["downsample_method"])

            ldsr_filename = st.session_state["uploaded_image"][i].name.split('.')[0] + '-gfpgan-ldsr2x'

            result.save(os.path.join(st.session_state["defaults"].post_processing.outdir_post_processing, f"{ldsr_filename}.png"))

        elif use_RealESRGAN and server_state["RealESRGAN"] is not None and use_GFPGAN and server_state["GFPGAN"] is not None:
            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].text("Running GFPGAN+RealESRGAN on image %d of %d..." % (i+1, len(st.session_state["uploaded_image"])))

            if "progress_bar" in st.session_state:
                st.session_state["progress_bar"].progress(
                    int(100 * float(i+1 if i+1 < len(st.session_state["uploaded_image"]) else len(st.session_state["uploaded_image"]))/float(len(st.session_state["uploaded_image"]))))

            torch_gc()
            cropped_faces, restored_faces, restored_img = server_state["GFPGAN"].enhance(st.session_state["uploaded_image"][i].pil_image, has_aligned=False, only_center_face=False, paste_back=True)
            gfpgan_sample = restored_img[:,:,::-1]

            if server_state["RealESRGAN"].model.name != realesrgan_model_name:
                #try_loading_RealESRGAN(realesrgan_model_name)
                load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

            output, img_mode = server_state["RealESRGAN"].enhance(gfpgan_sample[:,:,::-1])
            gfpgan_esrgan_filename = st.session_state["uploaded_image"][i].name.split('.')[0] + '-gfpgan-esrgan4x'
            gfpgan_esrgan_sample = output[:,:,::-1]
            gfpgan_esrgan_image = Image.fromarray(gfpgan_esrgan_sample)

            gfpgan_esrgan_image.save(os.path.join(st.session_state["defaults"].post_processing.outdir_post_processing, f"{gfpgan_esrgan_filename}.png"))



def layout():
    #st.info("Under Construction. :construction_worker:")
    st.session_state["progress_bar_text"] = st.empty()
    #st.session_state["progress_bar_text"].info("Nothing but crickets here, try generating something first.")

    st.session_state["progress_bar"] = st.empty()

    with st.form("post-processing-inputs"):
        # creating the page layout using columns
        col1, col2 = st.columns([1, 4], gap="medium")

        with col1:
            st.session_state["uploaded_image"] = st.file_uploader('Input Image', type=['png', 'jpg', 'jpeg', 'jfif', 'webp'], accept_multiple_files=True)


            # check if GFPGAN, RealESRGAN and LDSR are available.
            #if "GFPGAN_available" not in st.session_state:
            GFPGAN_available()

            #if "RealESRGAN_available" not in st.session_state:
            RealESRGAN_available()

            #if "LDSR_available" not in st.session_state:
            LDSR_available()

            if st.session_state["GFPGAN_available"] or st.session_state["RealESRGAN_available"] or st.session_state["LDSR_available"]:
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
                                                                            index=upscaling_method_list.index(st.session_state['defaults'].general.upscaling_method)
                                                                                if st.session_state['defaults'].general.upscaling_method in upscaling_method_list
                                                                                else 0)

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

                                st.session_state["ldsr_sampling_steps"] = st.number_input("Sampling Steps", value=st.session_state['defaults'].txt2img.LDSR_config.sampling_steps,
                                                                              help="")

                                st.session_state["preDownScale"] = st.number_input("PreDownScale", value=st.session_state['defaults'].txt2img.LDSR_config.preDownScale,
                                                                       help="")

                                st.session_state["postDownScale"] = st.number_input("postDownScale", value=st.session_state['defaults'].txt2img.LDSR_config.postDownScale,
                                                                       help="")

                                downsample_method_list = ['Nearest', 'Lanczos']
                                st.session_state["downsample_method"] = st.selectbox("Downsample Method", downsample_method_list,
                                                                                     index=downsample_method_list.index(st.session_state['defaults'].txt2img.LDSR_config.downsample_method))

                        else:
                            st.session_state["use_LDSR"] = False
                            st.session_state["LDSR_model"] = "model"

            #process = st.form_submit_button("Process Images", help="")

            #
            with st.expander("Output Settings", True):
                #st.session_state['defaults'].post_processing.save_original_images = st.checkbox("Save input images.", value=st.session_state['defaults'].post_processing.save_original_images,
                                                     #help="Save each original/input image next to the Post Processed image. "
                                                     #"This might be helpful for comparing the before and after images.")

                st.session_state['defaults'].post_processing.outdir_post_processing = st.text_input("Output Dir",value=st.session_state['defaults'].post_processing.outdir_post_processing,
                                                       help="Folder where the images will be saved after post processing.")

        with col2:
            st.subheader("Image")

            image_col1, image_col2, image_col3 = st.columns([2, 2, 2], gap="small")
            with image_col1:
                refresh = st.form_submit_button("Refresh", help='Refresh the image preview to show your uploaded image.')

            if st.session_state["uploaded_image"]:
                #print (type(st.session_state["uploaded_image"]))
                # if len(st.session_state["uploaded_image"]) == 1:
                st.session_state["input_image_preview"] = []
                st.session_state["input_image_caption"] = []
                st.session_state["output_image_preview"] = []
                st.session_state["output_image_caption"] = []
                st.session_state["input_image_preview_container"] = []
                st.session_state["prediction_table"] = []
                st.session_state["text_result"] = []

                for i in range(len(st.session_state["uploaded_image"])):
                    st.session_state["input_image_preview_container"].append(i)
                    st.session_state["input_image_preview_container"][i] = st.empty()

                    with st.session_state["input_image_preview_container"][i].container():
                        col1_output, col2_output, col3_output = st.columns([2, 2, 2], gap="medium")
                        with col1_output:
                            st.session_state["output_image_caption"].append(i)
                            st.session_state["output_image_caption"][i] = st.empty()
                            #st.session_state["output_image_caption"][i] = st.session_state["uploaded_image"][i].name

                            st.session_state["input_image_caption"].append(i)
                            st.session_state["input_image_caption"][i] = st.empty()
                            #st.session_state["input_image_caption"][i].caption(")

                            st.session_state["input_image_preview"].append(i)
                            st.session_state["input_image_preview"][i] = st.empty()
                            st.session_state["uploaded_image"][i].pil_image = Image.open(st.session_state["uploaded_image"][i]).convert('RGB')

                            st.session_state["input_image_preview"][i].image(st.session_state["uploaded_image"][i].pil_image, use_column_width=True, clamp=True)

                        with col2_output:
                            st.session_state["output_image_preview"].append(i)
                            st.session_state["output_image_preview"][i] = st.empty()

                            st.session_state["output_image_preview"][i].image(st.session_state["uploaded_image"][i].pil_image, use_column_width=True, clamp=True)

                    with st.session_state["input_image_preview_container"][i].container():

                        with col3_output:

                            #st.session_state["prediction_table"].append(i)
                            #st.session_state["prediction_table"][i] = st.empty()
                            #st.session_state["prediction_table"][i].table(pd.DataFrame(columns=["Model", "Filename", "Progress"]))

                            st.session_state["text_result"].append(i)
                            st.session_state["text_result"][i] = st.empty()
                            st.session_state["text_result"][i].code("", language="")

            #else:
                ##st.session_state["input_image_preview"].code('', language="")
                #st.image("images/streamlit/img2txt_placeholder.png", clamp=True)

        with image_col3:
            # Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
            process = st.form_submit_button("Process Images!")

        if process:
            with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders,index=[0]):
                #load_models(use_LDSR=st.session_state["use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
                            #use_GFPGAN=st.session_state["use_GFPGAN"], GFPGAN_model=st.session_state["GFPGAN_model"] ,
                            #use_RealESRGAN=st.session_state["use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"])

                if st.session_state["use_GFPGAN"]:
                    load_GFPGAN(model_name=st.session_state["GFPGAN_model"])

                if st.session_state["use_RealESRGAN"]:
                    load_RealESRGAN(st.session_state["RealESRGAN_model"])

                if st.session_state["use_LDSR"]:
                    load_LDSR(st.session_state["LDSR_model"])

            post_process(use_GFPGAN=st.session_state["use_GFPGAN"], GFPGAN_model=st.session_state["GFPGAN_model"],
                         use_RealESRGAN=st.session_state["use_RealESRGAN"], realesrgan_model_name=st.session_state["RealESRGAN_model"],
                         use_LDSR=st.session_state["use_LDSR"], LDSR_model_name=st.session_state["LDSR_model"])
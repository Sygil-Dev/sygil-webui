# base webui import and utils.
from webui_streamlit import st
from sd_utils import *

# streamlit imports
from streamlit import StopException
from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
from streamlit.elements import image as STImage

#other imports
import os
from typing import Union
from io import BytesIO
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

class plugin_info():
    plugname = "txt2img"
    description = "Text to Image"
    isTab = True
    displayPriority = 1


if os.path.exists(os.path.join(st.session_state['defaults'].general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
    GFPGAN_available = True
else:
    GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments","pretrained_models", f"{st.session_state['defaults'].general.RealESRGAN_model}.pth")):
    RealESRGAN_available = True
else:
    RealESRGAN_available = False	

#
def txt2img(prompt: str, ddim_steps: int, sampler_name: str, realesrgan_model_name: str,
            n_iter: int, batch_size: int, cfg_scale: float, seed: Union[int, str, None],
            height: int, width: int, separate_prompts:bool = False, normalize_prompt_weights:bool = True,
            save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
            save_as_jpg: bool = True, use_GFPGAN: bool = True, use_RealESRGAN: bool = True, 
            RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B", fp = None, variant_amount: float = None, 
            variant_seed: int = None, ddim_eta:float = 0.0, write_info_files:bool = True):

    outpath = st.session_state['defaults'].general.outdir_txt2img or st.session_state['defaults'].general.outdir or "outputs/txt2img-samples"

    seed = seed_to_int(seed)

    #prompt_matrix = 0 in toggles
    #normalize_prompt_weights = 1 in toggles
    #skip_save = 2 not in toggles
    #save_grid = 3 not in toggles
    #sort_samples = 4 in toggles
    #write_info_files = 5 in toggles
    #jpg_sample = 6 in toggles
    #use_GFPGAN = 7 in toggles
    #use_RealESRGAN = 8 in toggles

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(st.session_state["model"])
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(st.session_state["model"])
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(st.session_state["model"],'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(st.session_state["model"],'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(st.session_state["model"],'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(st.session_state["model"],'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(st.session_state["model"],'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(st.session_state["model"],'lms')
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
                use_RealESRGAN=st.session_state["use_RealESRGAN"],
                realesrgan_model_name=realesrgan_model_name,
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
            width = st.slider("Width:", min_value=64, max_value=4096, value=st.session_state['defaults'].txt2img.width, step=64)
            height = st.slider("Height:", min_value=64, max_value=4096, value=st.session_state['defaults'].txt2img.height, step=64)
            cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=st.session_state['defaults'].txt2img.cfg_scale, step=0.5, help="How strongly the image should follow the prompt.")
            seed = st.text_input("Seed:", value=st.session_state['defaults'].txt2img.seed, help=" The seed to use, if left blank a random seed will be generated.")
            batch_count = st.slider("Batch count.", min_value=1, max_value=100, value=st.session_state['defaults'].txt2img.batch_count, step=1, help="How many iterations or batches of images to generate in total.")

            bs_slider_max_value = 5
            if st.session_state.defaults.general.optimized:
                bs_slider_max_value = 100

            batch_size = st.slider(
                "Batch size",
                min_value=1,
                max_value=bs_slider_max_value,
                value=st.session_state.defaults.txt2img.batch_size,
                step=1,
                help="How many images are at once in a batch.\
                It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
                Default: 1")

            with st.expander("Preview Settings"):
                st.session_state["update_preview"] = st.checkbox("Update Image Preview", value=st.session_state['defaults'].txt2img.update_preview,
                                                                 help="If enabled the image preview will be updated during the generation instead of at the end. \
                 You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
                 By default this is enabled and the frequency is set to 1 step.")

                st.session_state["update_preview_frequency"] = st.text_input("Update Image Preview Frequency", value=st.session_state['defaults'].txt2img.update_preview_frequency,
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
            # If we have custom models available on the "models/custom" 
            #folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
            if st.session_state.CustomModel_available:
                st.session_state.custom_model = st.selectbox("Custom Model:", st.session_state.custom_models,
                                                                index=st.session_state["custom_models"].index(st.session_state['defaults'].general.default_model),
                                    help="Select the model you want to use. This option is only available if you have custom models \
                            on your 'models/custom' folder. The model name that will be shown here is the same as the name\
                            the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
                            will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4") 

            st.session_state.sampling_steps = st.slider("Sampling Steps",
            value=st.session_state['defaults'].txt2img.sampling_steps,
            min_value=st.session_state['defaults'].txt2img.slider_bounds.sampling.lower,
		    max_value=st.session_state['defaults'].txt2img.slider_bounds.sampling.upper,
		    step=st.session_state['defaults'].txt2img.slider_steps.sampling)

            sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
            sampler_name = st.selectbox("Sampling method", sampler_name_list,
                                        index=sampler_name_list.index(st.session_state['defaults'].txt2img.default_sampler), help="Sampling method to use. Default: k_euler")  



            #basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

            #with basic_tab:
                #summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
                    #help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")

            with st.expander("Advanced"):
                separate_prompts = st.checkbox("Create Prompt Matrix.", value=st.session_state['defaults'].txt2img.separate_prompts, help="Separate multiple prompts using the `|` character, and get all combinations of them.")
                normalize_prompt_weights = st.checkbox("Normalize Prompt Weights.", value=st.session_state['defaults'].txt2img.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
                save_individual_images = st.checkbox("Save individual images.", value=st.session_state['defaults'].txt2img.save_individual_images, help="Save each image generated before any filter or enhancement is applied.")
                save_grid = st.checkbox("Save grid",value=st.session_state['defaults'].txt2img.save_grid, help="Save a grid with all the images generated into a single image.")
                group_by_prompt = st.checkbox("Group results by prompt", value=st.session_state['defaults'].txt2img.group_by_prompt,
                                              help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
                write_info_files = st.checkbox("Write Info file", value=st.session_state['defaults'].txt2img.write_info_files, help="Save a file next to the image with informartion about the generation.")
                save_as_jpg = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].txt2img.save_as_jpg, help="Saves the images as jpg instead of png.")

                if st.session_state["GFPGAN_available"]:
                    st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].txt2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation.\
                            This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
                else:
                    st.session_state["use_GFPGAN"] = False

                if st.session_state["RealESRGAN_available"]:
                    st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=st.session_state['defaults'].txt2img.use_RealESRGAN,
                                                                     help="Uses the RealESRGAN model to upscale the images after the generation.\
                            This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
                    st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)  
                else:
                    st.session_state["use_RealESRGAN"] = False
                    st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"

                variant_amount = st.slider("Variant Amount:", value=st.session_state['defaults'].txt2img.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
                variant_seed = st.text_input("Variant Seed:", value=st.session_state['defaults'].txt2img.seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")
        galleryCont = st.empty()

        if generate_button:
            #print("Loading models")
            # load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.	
            load_models(False, st.session_state["use_GFPGAN"], st.session_state["use_RealESRGAN"], st.session_state["RealESRGAN_model"], st.session_state["CustomModel_available"],
                        st.session_state["custom_model"])  
            

            try:
                #
                output_images, seeds, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, st.session_state["RealESRGAN_model"], batch_count, batch_size,
                                                            cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
                                                            save_grid, group_by_prompt, save_as_jpg, st.session_state["use_GFPGAN"], st.session_state["use_RealESRGAN"], st.session_state["RealESRGAN_model"],
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
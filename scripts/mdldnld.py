import os
import streamlit as st
from pathlib import Path

dest = "/pebble_tmp/models/stable_diffusion_models"

def exec_cmd(prompt):
    res = os.popen(prompt)
    output = res.readlines()
    print(output)

    return output

def print_progress(msg, strmlit_ui):
    st.write(msg) if strmlit_ui else print(msg)

def dwnld_model(exceptions_log, address, target_location, source_model_name=None, target_model_name=None, strmlit_ui=False):
    target_name = Path(target_model_name)

    if not target_name.exists():
        msg = f"Downloading {address} to {target_location}"
        print_progress(msg, strmlit_ui)
        try:
            os.system(f'wget -bqc {address} -P {target_location}')
            if target_model_name is not None:
                os.system(f'mv {source_model_name} {target_model_name}')
            print_progress("Done", strmlit_ui)
        except Exception as e:
            st.write(e) if strmlit_ui else exceptions_log.append([msg,e])
    else:
        msg = f"Skipping {target_name}"
        print_progress(msg, strmlit_ui)
    
    return exceptions_log

strmlit_ui = True
class Models():
    exceptions_log = []
    dest = dest.strip("/").replace(" ", "_")

    dest = os.path.expanduser("~/" + dest)

    def modelSD():
        exceptions_log = []
        exceptions_log = dwnld_model(exceptions_log, 
                                    address = 'https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media', 
                                    target_location = f'{dest}/models/ldm/stable-diffusion-v1/', 
                                    source_model_name= f'{dest}/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt?alt=media', 
                                    target_model_name= f'{dest}/models/ldm/stable-diffusion-v1/model.ckpt', 
                                    strmlit_ui=True)

    def realESRGAN():
        exceptions_log = []
        exceptions_log = dwnld_model(exceptions_log, 
                                    address = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
                                    target_location = f'{dest}/models/src/realesrgan/experiments/pretrained_models', 
                                    source_model_name= f'{dest}/models/src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus.pth', 
                                    target_model_name= f'{dest}/models/src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus.pth', 
                                    strmlit_ui=True)
        
        exceptions_log = dwnld_model(exceptions_log, 
                                    address = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', 
                                    target_location = f'{dest}/models/src/realesrgan/experiments/pretrained_models', 
                                    source_model_name= f'{dest}/models/src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth', 
                                    target_model_name= f'{dest}/models/src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth', 
                                    strmlit_ui=True)
    
    def GFPGAN():
        exceptions_log = []
        exceptions_log = dwnld_model(exceptions_log, 
                                    address = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', 
                                    target_location = f'{dest}/models/src/gfpgan/experiments/pretrained_models', 
                                    source_model_name= f'{dest}/models/src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth', 
                                    target_model_name= f'{dest}/models/src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth', 
                                    strmlit_ui=True)
    
        
                                    
    def modelLD():
        if not Path(f'{dest}/models/src/latent-diffusion').exists():
            os.system(f'cd {dest}/models/ ; git clone https://github.com/devilismyfriend/latent-diffusion.git')
            os.system(f'mv {dest}/models/latent-diffusion  {dest}/models/src/latent-diffusion')
        exceptions_log = [] 
        exceptions_log = dwnld_model(exceptions_log, 
                                    address = 'https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1', 
                                    target_location = f'{dest}/models/src/latent-diffusion/experiments/pretrained_models', 
                                    source_model_name= f'{dest}/models/src/latent-diffusion/experiments/pretrained_models/index.html?dl=1', 
                                    target_model_name= f'{dest}/models/src/latent-diffusion/experiments/pretrained_models/project.yaml', 
                                    strmlit_ui=True)

        exceptions_log = dwnld_model(exceptions_log, 
                                    address = 'https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1', 
                                    target_location = f'{dest}/models/src/latent-diffusion/experiments/pretrained_models', 
                                    source_model_name= f'{dest}/models/src/latent-diffusion/experiments/pretrained_models/index.html?dl=1', 
                                    target_model_name= f'{dest}/models/src/latent-diffusion/experiments/pretrained_models/model.ckpt', 
                                    strmlit_ui=True)
    

    # Stable Diffusion Conecpt Library
    def SD_conLib():

        if not Path(f'{dest}/models/custom/sd-concepts-library').exists():
            os.system(
                f'cd {dest}/models/ ; git clone https://github.com/sd-webui/sd-concepts-library custom/')


    # Blip Model
    def modelBlip():
        exceptions_log = [] 

        if not Path(f'{dest}/models/custom/blip/model__base_caption.pth').exists():
            # return st.write(f"Blip Model is to be installed !")
            exceptions_log = dwnld_model(exceptions_log, 
                                    address = "https://cdn-lfs.huggingface.co/repos/cd/15/cd1551e1e53c5049819b5349e3e386c497a767dfeebb8e146ae2adb8f39c8d10/96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086?response-content-disposition=attachment%3B%20filename%3D%22model__base_caption.pth%22", 
                                    target_location = f'{dest}/models/custom/blip/', 
                                    source_model_name= f'{dest}/models/custom/blip/96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086?response-content-disposition=attachment;\ filename=\"model__base_caption.pth\"', 
                                    target_model_name= f'{dest}/models/custom/blip/model__base_caption.pth', 
                                    strmlit_ui=True)
            # os.system(f"wget -O {dest}/models/custom/blip/model__base_caption.pth https://cdn-lfs.huggingface.co/repos/cd/15/cd1551e1e53c5049819b5349e3e386c497a767dfeebb8e146ae2adb8f39c8d10/96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086?response-content-disposition=attachment%3B%20filename%3D%22model__base_caption.pth%22")


    # Waifu Diffusion v1.2
    def modelWD():

        if not Path(f'{dest}/models/custom/waifu-diffusion').exists():
            os.system(
                f"cd {dest}/models/ ;git clone https://huggingface.co/hakurei/waifu-diffusion custom/waifu-diffusion")


    # Waifu Diffusion v1.2 Pruned
    def modelWDP():

        if not Path(f'{dest}/models/custom/pruned-waifu-diffusion').exists():
            os.system(
                f"cd {dest}/models/ ;git clone https://huggingface.co/crumb/pruned-waifu-diffusion custom/pruned-waifu-diffusion")

    # TrinArt Stable Diffusion v2
    def modelTSD():

        if not Path(f'{dest}/models/custom/trinart_stable_diffusion_v2').exists():
            os.system(
                f"cd {dest}/models/ ;git clone https://huggingface.co/naclbit/trinart_stable_diffusion_v2 custom/trinart_stable_diffusion_v2")



def st_ui():
    st.title("Stable Diffusion models download")
    with st.spinner("Stable Diffusion"):
        Models.modelSD()

    with st.spinner("RealESRGAN"):
        Models.realESRGAN()

    with st.spinner("GFPGAN"):
        Models.GFPGAN()

    with st.spinner("Latent Diffusion"):
        Models.modelLD()

    with st.spinner("SD Concept Lib"):
        Models.SD_conLib()

    with st.spinner("Blip"):
        Models.modelBlip()

    with st.spinner("Waifu Diffusion"):
        Models.modelWD()

    with st.spinner("Waifu Pruned"):
        Models.modelLD()

    with st.spinner("TrinArt SD"):
        Models.modelTSD()

if __name__ == "__main__":
    st_ui()
    

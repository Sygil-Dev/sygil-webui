# base webui import and utils.
from webui_streamlit import st
from sd_utils import *

# streamlit imports


#other imports
import pandas as pd
from io import StringIO

# Temp imports 


# end of imports
#---------------------------------------------------------------------------------------------------------------

def layout():
    #search = st.text_input(label="Search", placeholder="Type the name of the model you want to search for.", help="")

    csvString = f"""
                    ,Stable Diffusion v1.4            , ./models/ldm/stable-diffusion-v1               , https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media                  
                    ,GFPGAN v1.3                      , ./src/gfpgan/experiments/pretrained_models     , https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth                     
                    ,RealESRGAN_x4plus                , ./src/realesrgan/experiments/pretrained_models , https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth            
                    ,RealESRGAN_x4plus_anime_6B       , ./src/realesrgan/experiments/pretrained_models , https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth 
                    ,Waifu Diffusion v1.2             , ./models/custom                                , http://wd.links.sd:8880/wd-v1-2-full-ema.ckpt
                    ,TrinArt Stable Diffusion v2      , ./models/custom                                , https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step115000.ckpt
                    ,Stable Diffusion Concept Library , ./models/customsd-concepts-library             , https://github.com/sd-webui/sd-concepts-library
                    """
    colms = st.columns((1, 3, 5, 5))
    columns = ["№",'Model Name','Save Location','Download Link']

    # Convert String into StringIO
    csvStringIO = StringIO(csvString)
    df = pd.read_csv(csvStringIO, sep=",", header=None, names=columns)		

    for col, field_name in zip(colms, columns):
        # table header
        col.write(field_name)

    for x, model_name in enumerate(df["Model Name"]):
        col1, col2, col3, col4 = st.columns((1, 3, 4, 6))
        col1.write(x)  # index
        col2.write(df['Model Name'][x])
        col3.write(df['Save Location'][x])
        col4.write(df['Download Link'][x])    
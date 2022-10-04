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
import wget

# Temp imports 


# end of imports
#---------------------------------------------------------------------------------------------------------------
# (.+)\t(.+)\t(.+)\t(.+)
# Stable Diffusion v1-4	model.ckpt	models/ldm/stable-diffusion-v1	https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media
# GFPGAN	GFPGANv1.3.pth	models/gfpgan	https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
# RealESRGAN	RealESRGAN_x4plus.pth	models/realesrgan	https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
# RealESRGAN	RealESRGAN_x4plus_anime_6B.pth	models/realesrgan	https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
# LDSR	project.yaml	models/latent-diffusion	https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1
# LDSR	model.ckpt	models/latent-diffusion	https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1
# Waifu Diffusion	waifu-diffusion.ckpt	models/custom	https://huggingface.co/crumb/pruned-waifu-diffusion/resolve/main/model-pruned.ckpt
# TrinArt	trinart.ckpt	models/custom	https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt
# BLIP	model__base_caption.pth	models/blip	https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth
# ViT-L-14	pytorch_model.bin	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
# ViT-L-14	config.json	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json
# ViT-L-14	merges.txt	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt
# ViT-L-14	preprocessor_config.json	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json
# ViT-L-14	special_tokens_map.json	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/special_tokens_map.json
# ViT-L-14	tokenizer.json	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json
# ViT-L-14	tokenizer_config.json	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer_config.json
# ViT-L-14	vocab.json	models/clip-vit-large-patch14	https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json
# array of models
# {'model_name': \1, 'files': [{'file_name': \2, 'file_path': \2, 'file_url': \3}]}

model_list = [
    {'model_name': 'Stable Diffusion v1-4', 'files': [{'file_name': 'model.ckpt', 'file_path': 'models/ldm/stable-diffusion-v1', 'file_url': 'https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media'}]},
    {'model_name': 'GFPGAN', 'files': [{'file_name': 'GFPGANv1.4.pth', 'file_path': 'models/gfpgan', 'file_url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'}]},
    {'model_name': 'RealESRGAN', 'files': 
    [{'file_name': 'RealESRGAN_x4plus.pth', 'file_path': 'models/realesrgan', 'file_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'},
        {'file_name': 'RealESRGAN_x4plus_anime_6B.pth', 'file_path': 'models/realesrgan', 'file_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'}
    ]},
    {'model_name': 'LDSR', 'files': [
        {'file_name': 'project.yaml', 'file_path': 'models/latent-diffusion', 'file_url': 'https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1'},
        {'file_name': 'model.ckpt', 'file_path': 'models/latent-diffusion', 'file_url': 'https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1'}
    ]},
    {'model_name': 'Waifu Diffusion', 'files': [{'file_name': 'waifu-diffusion.ckpt', 'file_path': 'models/custom', 'file_url': 'https://huggingface.co/crumb/pruned-waifu-diffusion/resolve/main/model-pruned.ckpt'}]},
    {'model_name': 'TrinArt', 'files': [{'file_name': 'trinart.ckpt', 'file_path': 'models/custom', 'file_url': 'https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt'}]},
    {'model_name': 'BLIP', 'files': [{'file_name': 'model__base_caption.pth', 'file_path': 'models/blip', 'file_url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'}]},
    {'model_name': 'ViT-L-14', 'files': [
        {'file_name': 'pytorch_model.bin', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin'},
        {'file_name': 'config.json', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json'},
        {'file_name': 'merges.txt', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt'},
        {'file_name': 'preprocessor_config.json', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json'},
        {'file_name': 'special_tokens_map.json', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/special_tokens_map.json'},
        {'file_name': 'tokenizer.json', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json'},
        {'file_name': 'tokenizer_config.json', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer_config.json'},
        {'file_name': 'vocab.json', 'file_path': 'models/clip-vit-large-patch14', 'file_url': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json'}
    ]}
]

def download_file(file_name, file_path, file_url):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(file_path + '/' + file_name):
        print('Downloading ' + file_name + '...')
        # TODO - add progress bar in streamlit
        wget.download(url=file_url, out=file_path + '/' + file_name)
    else:
        print(file_name + ' already exists.')

def download_model(model_name):
    """ Download all files from model_list[model_name] """
    for file in model_list[model_name]:
        download_file(file['file_name'], file['file_path'], file['file_url'])
    return



def layout():
    st.title("Model Manager") 
    

    # model in model list display model_name (unique) and files (list of dicts) st.columns
    # and for each file in files check if file exists in file_path and if not display download button
    model_name, files, download_button = st.columns(3)
    for model in model_list:
        with model_name:
            st.write(model['model_name'])
        with files:
            files_exist = 0
            for file in model['files']:
                if os.path.exists(file['file_path'] + '/' + file['file_name']):
                    files_exist += 1
            st.write('✅' if files_exist == len(model['files']) else str(files_exist) + '/' + str(len(model['files'])) + '❌')
        with download_button:
            files_needed = []
            for file in model['files']:
                if not os.path.exists(file['file_path'] + '/' + file['file_name']):
                    files_needed.append(file)
            if len(files_needed) > 0:
                if st.button('Download', key=model['model_name'], help='Download ' + model['model_name']):
                    for file in files_needed:
                        download_file(file['file_name'], file['file_path'], file['file_url'])
                else:
                    st.empty()
            else:
                st.write('✅')


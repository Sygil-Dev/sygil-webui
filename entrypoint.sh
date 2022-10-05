#!/bin/bash
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
#
# Starts the webserver inside the docker container
#

# set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
export PYTHONPATH=$SCRIPT_DIR

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
    echo "SSH Service Started"
fi


MODEL_DIR="${SCRIPT_DIR}/user_data/model_cache"
mkdir -p $MODEL_DIR
# Array of model files to pre-download
# local filename
# local path in container (no trailing slash)
# download URL
# sha256sum
MODEL_FILES=(
    'model.ckpt models/ldm/stable-diffusion-v1 https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media'
    'GFPGANv1.4.pth models/gfpgan https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
    'detection_Resnet50_Final.pth gfpgan/weights https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
    'parsing_parsenet.pth gfpgan/weights https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    'RealESRGAN_x4plus.pth models/realesrgan https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    'RealESRGAN_x4plus_anime_6B.pth models/realesrgan https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    'project.yaml models/ldsr https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1'
    'model.ckpt models/ldsr https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1'
    'waifu-diffusion.ckpt models/custom https://huggingface.co/crumb/pruned-waifu-diffusion/resolve/main/model-pruned.ckpt'
    'trinart.ckpt models/custom https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt'
    'model__base_caption.pth models/blip https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    'pytorch_model.bin models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin'
    'config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json'
    'merges.txt models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt'
    'preprocessor_config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json'
    'special_tokens_map.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/special_tokens_map.json'
    'tokenizer.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json'
    'tokenizer_config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer_config.json'
    'vocab.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json'
)

downloadModel() {
    local file=$1
    local path="${SCRIPT_DIR}/${2}"
    local path_dir="${MODEL_DIR}/$2"
    local url=$3

    if [[ ! -e "${MODEL_DIR}/$2/${file}" ]]; then
        echo "Downloading: ${url} please wait..."
        mkdir -p ${MODEL_DIR}/$2
        mkdir -p ${path}
        wget --output-document=${MODEL_DIR}/$2/${file} --no-verbose --show-progress --progress=dot:giga ${url}
        ln -sf ${MODEL_DIR}/$2/${file} ${path}/${file}
        if [[ -e "${path}/${file}" ]]; then
            echo "saved ${file}"
        else
            echo "error saving ${MODEL_DIR}/$2/${file}!"
            exit 1
        fi
    fi
}

echo "Downloading model files..."
for models in "${MODEL_FILES[@]}"; do
    model=($models)
    if [[ ! -e ${model[1]}/${model[0]} || ! -L ${model[1]}/${model[0]} ]]; then
        downloadModel ${model[0]} ${model[1]} ${model[2]}
    fi
done

# Create directory for diffusers models
mkdir -p ${MODEL_DIR}/diffusers/stable-diffusion-v1-4
mkdir -p ${MODEL_DIR}/diffusers/waifu-diffusion
mkdir -p ${SCRIPT_DIR}/diffusers/stable-diffusion-v1-4
mkdir -p ${SCRIPT_DIR}/diffusers/waifu-diffusion
# Link tokenizer to diffusers models
ln -fs ${SCRIPT_DIR}/models/clip-vit-large-patch14/ ${SCRIPT_DIR}/diffusers/stable-diffusion-v1-4/tokenizer
ln -fs ${SCRIPT_DIR}/models/clip-vit-large-patch14/ ${SCRIPT_DIR}/diffusers/waifu-diffusion/tokenizer

if [[ -e "${MODEL_DIR}/sd-concepts-library" ]]; then
    # concept library exists, update
    cd ${MODEL_DIR}/sd-concepts-library
    git pull
else
    # concept library does not exist, clone
    cd ${MODEL_DIR}
    git clone https://github.com/sd-webui/sd-concepts-library.git
fi
# create directory and link concepts library
mkdir -p ${SCRIPT_DIR}/models/custom
ln -fs ${MODEL_DIR}/sd-concepts-library/sd-concepts-library/ ${SCRIPT_DIR}/models/custom/sd-concepts-library

mkdir -p ${SCRIPT_DIR}/user_data/outputs
ln -fs ${SCRIPT_DIR}/user_data/outputs/ ${SCRIPT_DIR}/outputs

echo "export HF_HOME=${MODEL_DIR}" >> ~/.bashrc
echo "export XDG_CACHE_HOME=${MODEL_DIR}" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=${MODEL_DIR}" >> ~/.bashrc
source ~/.bashrc
cd $SCRIPT_DIR
launch_command="streamlit run ${SCRIPT_DIR}/scripts/webui_streamlit.py"

$launch_command

sleep infinity

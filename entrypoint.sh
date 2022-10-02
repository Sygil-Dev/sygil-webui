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
    'model.ckpt models/ldm/stable-diffusion-v1 https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
    'GFPGANv1.3.pth src/gfpgan/experiments/pretrained_models https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70'
    'RealESRGAN_x4plus.pth src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
    'RealESRGAN_x4plus_anime_6B.pth src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
    'project.yaml src/latent-diffusion/experiments/pretrained_models https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1 9d6ad53c5dafeb07200fb712db14b813b527edd262bc80ea136777bdb41be2ba'
    'model.ckpt src/latent-diffusion/experiments/pretrained_models https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1 c209caecac2f97b4bb8f4d726b70ac2ac9b35904b7fc99801e1f5e61f9210c13'
    'waifu-diffusion.ckpt models/custom https://huggingface.co/crumb/pruned-waifu-diffusion/resolve/main/model-pruned.ckpt 9b31355f90fea9933847175d4731a033f49f861395addc7e153f480551a24c25'
    'trinart.ckpt models/custom https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt c1799d22a355ba25c9ceeb6e3c91fc61788c8e274b73508ae8a15877c5dbcf63'
    'model__base_caption.pth models/blip https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth 96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086'
    'pytorch_model.bin models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin f1a17cdbe0f36fec524f5cafb1c261ea3bbbc13e346e0f74fc9eb0460dedd0d3'
    'config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json 8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a'
    'merges.txt models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt 9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a'
    'preprocessor_config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json 910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd'
    'special_tokens_map.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/special_tokens_map.json f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf'
    'tokenizer.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4'
    'tokenizer_config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer_config.json deef455e52fa5e8151e339add0582e4235f066009601360999d3a9cda83b1129'
    'vocab.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json 3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69'
)


# Function to checks for valid hash for model files and download/replaces if invalid or does not exist
validateDownloadModel() {
    local file=$1
    local path="${SCRIPT_DIR}/${2}"
    local url=$3
    local hash=$4

    echo "checking ${file}..."
    sha256sum --check --status <<< "${hash} ${MODEL_DIR}/${file}.${hash}"
    if [[ $? == "1" ]]; then
        echo "Downloading: ${url} please wait..."
        mkdir -p ${path}
        wget --output-document=${MODEL_DIR}/${file}.${hash} --no-verbose --show-progress --progress=dot:giga ${url}
        ln -sf ${MODEL_DIR}/${file}.${hash} ${path}/${file}
        if [[ -e "${path}/${file}" ]]; then
            echo "saved ${file}"
        else
            echo "error saving ${path}/${file}!"
            exit 1
        fi
    else
        if [[ ! -e ${path}/${file} || ! -L ${path}/${file} ]]; then
            mkdir -p ${path}
            ln -sf ${MODEL_DIR}/${file}.${hash} ${path}/${file}
            echo -e "linked valid ${file}\n"
        else
            echo -e "${file} is valid!\n"
        fi
    fi
}


# Validate model files
if [ $VALIDATE_MODELS == "false" ]; then
    echo "Skipping model file validation..."
else
    echo "Validating model files..."
    for models in "${MODEL_FILES[@]}"; do
        model=($models)
        if [[ ! -e ${model[1]}/${model[0]} || ! -L ${model[1]}/${model[0]} || -z $VALIDATE_MODELS || $VALIDATE_MODELS == "true" ]]; then
            validateDownloadModel ${model[0]} ${model[1]} ${model[2]} ${model[3]}
        fi
    done
    mkdir -p ${MODEL_DIR}/stable-diffusion-v1-4
    mkdir -p ${MODEL_DIR}/waifu-diffusion

    ln -fs ${SCRIPT_DIR}/models/clip-vit-large-patch14/ ${MODEL_DIR}/stable-diffusion-v1-4/tokenizer
    ln -fs ${SCRIPT_DIR}/models/clip-vit-large-patch14/ ${MODEL_DIR}/waifu-diffusion/tokenizer
fi

if [[ -e "${MODEL_DIR}/sd-concepts-library" ]]; then
    cd ${MODEL_DIR}/sd-concepts-library
    git pull
else
    cd ${MODEL_DIR}
    git clone https://github.com/sd-webui/sd-concepts-library
fi
mkdir -p ${SCRIPT_DIR}/models/custom
ln -fs ${MODEL_DIR}/sd-concepts-library/sd-concepts-library ${SCRIPT_DIR}/models/custom

echo "export HF_HOME=${MODEL_DIR}" >> ~/.bashrc
echo "export XDG_CACHE_HOME=${MODEL_DIR}" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=${MODEL_DIR}" >> ~/.bashrc
source ~/.bashrc
cd $SCRIPT_DIR
launch_command="streamlit run ${SCRIPT_DIR}/scripts/webui_streamlit.py"

$launch_command

sleep infinity

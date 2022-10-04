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
# b3sum
MODEL_FILES=(
    'model.ckpt models/ldm/stable-diffusion-v1 https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media 5a4792c52c98aaaf7ce8943b6d0700c217e7bb163857479bfea3ec56de338377'
    'GFPGANv1.3.pth src/gfpgan/experiments/pretrained_models https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth 890232331ae7c81282c7998a474e4affc6411b6874c9951cf4d22d591dfaa77b'
    'RealESRGAN_x4plus.pth src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth 37d8f128b58ce8cd4f2205058ca0069567e989ea45ac3b9131472c96a4770692'
    'RealESRGAN_x4plus_anime_6B.pth src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth 717c1bcb17218786f29dd5377dd53a905fb5ec33c6ca12cb8dc0e3f2f18fa1b7'
    'project.yaml src/latent-diffusion/experiments/pretrained_models https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1 9b519a1d36742c03f871be7bb86c835721726bb21779669b16f284bfa7288129'
    'model.ckpt src/latent-diffusion/experiments/pretrained_models https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1 0f663e3e2d8acf08caa7f8b9b77ea6b528de7c889b355502aae03e4dd6a8bd9e'
    'waifu-diffusion.ckpt models/custom https://huggingface.co/crumb/pruned-waifu-diffusion/resolve/main/model-pruned.ckpt 5a0b97dbd1ae2b4af027a62942a0ef09cf22ecc6b80fdf0ede1d5d9bbced2553'
    'trinart.ckpt models/custom https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt c8a01f56cebca2dc288f14bca9b1a85be72c7a1b33d8c231d9782b3dd2de33ec'
    'model__base_caption.pth models/blip https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth 1b273d45faa24314653131d9652768c0f46ae06ffd3cf6827551ab02cafa192f'
    'pytorch_model.bin models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin af5cfe5481ec844bd56a751163d9fcea733c4b16c9627803ef829b21fe64c86a'
    'config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json 256ddac342ff5c42435ab6a25dc7fd1973cd44a892ee1e6e2a757e0eace90fa4'
    'merges.txt models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt 64a704d74f18e52434c694dac7226f4dfa267600c615929126c4795be0d3d8d9'
    'preprocessor_config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json 2a7717a5dfbac23cd17e2610844c3d87631599a842609c2fbc9ba5d01573c2e2'
    'special_tokens_map.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/special_tokens_map.json 619ae5e5cb385bdf16bf29f0151a78ec684ecf714ef3c152f7d178bc1f07634c'
    'tokenizer.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json 8da70f386debb2f514a23df2838c10f983d8fecb86459cc0afce594c8f6efd33'
    'tokenizer_config.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer_config.json 524be81c6c0c821214c5e06aec9acf6a4723bc8ce1c812b4f89624ea04964b3d'
    'vocab.json models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json 1933eae25daadf9ce6ecbf884826181549a76817d3b5fcbd5cf6cb5baf8935c5'
)


# Function to checks for valid hash for model files and download/replaces if invalid or does not exist
validateDownloadModel() {
    local file=$1
    local path="${SCRIPT_DIR}/${2}"
    local url=$3
    local hash=$4

    echo "checking ${file}..."
    b3sum --check --quiet <<< "${hash}  ${MODEL_DIR}/${file}.${hash}"
    if [[ $? == "1" ]]; then
        echo "Downloading: ${url} please wait..."
        mkdir -p ${path}
        wget --output-document=${MODEL_DIR}/${file}.${hash} --no-verbose --show-progress --progress=dot:giga ${url}
        ln -sf ${MODEL_DIR}/${file}.${hash} ${path}/${file}
        if [[ -e "${path}/${file}" ]]; then
            echo "checking ${file}..."
            b3sum --check --quiet <<< "${hash}  ${MODEL_DIR}/${file}.${hash}"
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

#!/bin/bash -i

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
# Start the Stable Diffusion WebUI for Linux Users

DIRECTORY="."
ENV_FILE="environment.yaml"
ENV_NAME="ldm"
ENV_MODIFIED=$(date -r $ENV_FILE "+%s")
ENV_MODIFED_FILE=".env_updated"
ENV_UPDATED=0

INSTALL_ENV_DIR="$(pwd)/../installer_files/env" # since linux-sd.sh clones the repo into a subfolder
if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

# Models used for upscaling
GFPGAN_MODEL="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
LATENT_DIFFUSION_REPO="https://github.com/devilismyfriend/latent-diffusion.git"
LSDR_CONFIG="https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
LSDR_MODEL="https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
REALESRGAN_MODEL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
REALESRGAN_ANIME_MODEL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
SD_CONCEPT_REPO="https://github.com/Sygil-Dev/sd-concepts-library/archive/refs/heads/main.zip"


if [[ -f $ENV_MODIFED_FILE ]]; then 
    ENV_MODIFIED_CACHED=$(<${ENV_MODIFED_FILE})
else 
    ENV_MODIFIED_CACHED=0
fi

# Setup the Conda env for the project. This will also handle updating the env as needed too.
conda_env_setup () {
    # Set conda path if it is not already in default environment
    CUSTOM_CONDA_PATH=

    # Allow setting custom path via file to allow updates of this script without undoing custom path
    if [ -f custom-conda-path.txt ]; then
        CUSTOM_CONDA_PATH=$(cat custom-conda-path.txt)
    fi

    # If a custom conda isn't specified, and the installer downloaded conda for the user, then use that
    if [ -f "$INSTALL_ENV_DIR/etc/profile.d/conda.sh" ] && [ "$CUSTOM_CONDA_PATH" == "" ]; then
        . "$INSTALL_ENV_DIR/etc/profile.d/conda.sh"
    fi

    # If custom path is set above, try to setup conda environment
    if [ -f "${CUSTOM_CONDA_PATH}/etc/profile.d/conda.sh" ]; then
        . "${CUSTOM_CONDA_PATH}/etc/profile.d/conda.sh"
    elif [ -n "${CUSTOM_CONDA_PATH}" ] && [ -f "${CUSTOM_CONDA_PATH}/bin" ]; then
        export PATH="${CUSTOM_CONDA_PATH}/bin:$PATH"
    fi

    if ! command -v conda >/dev/null; then
        printf "Anaconda3 not found. Install from here https://www.anaconda.com/products/distribution\n"
        exit 1
    fi

    # Create/update conda env if needed
    if ! conda env list | grep ".*${ENV_NAME}.*" >/dev/null 2>&1; then
        printf "Could not find conda env: ${ENV_NAME} ... creating ... \n\n"
        conda env create -f $ENV_FILE
        ENV_UPDATED=1
    elif [[ ! -z $CONDA_FORCE_UPDATE && $CONDA_FORCE_UPDATE == "true" ]] || (( $ENV_MODIFIED > $ENV_MODIFIED_CACHED )); then
        printf "Updating conda env: ${ENV_NAME} ...\n\n"
        PIP_EXISTS_ACTION=w conda env update --file $ENV_FILE --prune
        ENV_UPDATED=1
    fi

    # Clear artifacts from conda after create/update
    if (( $ENV_UPDATED > 0 )); then
        conda clean --all
        echo -n $ENV_MODIFIED > $ENV_MODIFED_FILE
    fi
}

# Activate conda environment
conda_env_activation () {
    conda activate $ENV_NAME
    conda info | grep active
}

# Checks to see if the upscaling models exist in their correct locations. If they do not they will be downloaded as required
post_processor_model_loading () {
    # Check to see if GFPGAN has been added yet, if not it will download it and place it in the proper directory
    if [ -f "$DIRECTORY/models/gfpgan/GFPGANv1.3.pth" ]; then
        printf "GFPGAN already exists. Continuing...\n\n"
    else
        printf "Downloading GFPGAN model. Please wait...\n"
        wget $GFPGAN_MODEL -P $DIRECTORY/models/gfpgan
    fi

    # Check to see if realESRGAN has been added yet, if not it will download it and place it in the proper directory
    if [ -f "$DIRECTORY/models/realesrgan/RealESRGAN_x4plus.pth" ]; then
        printf "realESRGAN already exists. Continuing...\n\n"
    else
        printf "Downloading realESRGAN model. Please wait...\n"
        wget $REALESRGAN_MODEL -P $DIRECTORY/models/realesrgan
        wget $REALESRGAN_ANIME_MODEL -P $DIRECTORY/models/realesrgan
    fi

    # Check to see if LDSR has been added yet, if not it will be cloned and its models downloaded to the correct directory
    if [ -f "$DIRECTORY/models/ldsr/model.ckpt" ]; then
        printf "LDSR already exists. Continuing...\n\n"
    else
        printf "Cloning LDSR and downloading model. Please wait...\n"
        git clone $LATENT_DIFFUSION_REPO
        mv latent-diffusion $DIRECTORY/models/ldsr
        mkdir $DIRECTORY/models/ldsr/experiments
        mkdir $DIRECTORY/models/ldsr
        wget $LSDR_CONFIG -P $DIRECTORY/models/ldsr
        mv $DIRECTORY/models/ldsr/index.html?dl=1 $DIRECTORY/models/ldsr/project.yaml
        wget $LSDR_MODEL -P $DIRECTORY/models/ldsr
        mv $DIRECTORY/models/ldsr/index.html?dl=1 $DIRECTORY/models/ldsr/model.ckpt
    fi

    # Check to see if SD Concepts has been added yet, if not it will download it and place it in the proper directory
    if [ -d "$DIRECTORY/models/custom/sd-concepts-library" ]; then
        printf "SD Concepts Library already exists. Continuing...\n\n"
    else
        printf "Downloading and Extracting SD Concepts Library model. Please wait...\n"
        mkdir $DIRECTORY/models/custom
        wget $SD_CONCEPT_REPO
        if ! command -v unzip &> /dev/null
        then
            printf "Warning: unzip could not be found. \nPlease install 'unzip' from your package manager and rerun this program.\n"
            exit 1
        fi
        unzip main.zip
        mv sd-concepts-library-main/sd-concepts-library $DIRECTORY/models/custom
    fi
}

# Show the user a prompt asking them which version of the WebUI they wish to use, Streamlit or Gradio
launch_webui () {
    # skip the prompt if --bridge command-line argument is detected
    for arg in "$@"; do
        if [ "$arg" == "--bridge" ]; then
           python -u scripts/relauncher.py "$@"
           return
        fi
    done
    printf "\n\n########## LAUNCH USING STREAMLIT OR GRADIO? ##########\n\n"
    printf "Do you wish to run the WebUI using the Gradio or StreamLit Interface?\n\n"
    printf "Streamlit: \nHas A More Modern UI \nMore Features Planned \nWill Be The Main UI Going Forward \nCurrently In Active Development \nMissing Some Gradio Features\n\n"
    printf "Gradio: \nCurrently Feature Complete \nUses An Older Interface Style \nWill Not Receive Major Updates\n\n"
    printf "Which Version of the WebUI Interface do you wish to use?\n"
    select yn in "Streamlit" "Gradio"; do
        case $yn in
            Streamlit ) printf "\nStarting Stable Diffusion WebUI: Streamlit Interface. Please Wait...\n"; python -m streamlit run scripts/webui_streamlit.py --theme.base dark --server.address localhost; break;;
            Gradio ) printf "\nStarting Stable Diffusion WebUI: Gradio Interface. Please Wait...\n"; python scripts/relauncher.py "$@"; break;;
        esac
    done
}

# Function to initialize the other functions
start_initialization () {
    conda_env_setup
    sd_model_loading
    post_processor_model_loading
    conda_env_activation
    if [ ! -e "models/ldm/stable-diffusion-v1/*.ckpt" ]; then
        echo "Your model file does not exist! Streamlit will handle this automatically, however Gradio still requires this file be placed manually. If you intend to use the Gradio interface, place it in 'models/ldm/stable-diffusion-v1' with the name 'model.ckpt'."
        read -p "Once you have sd-v1-4.ckpt in the project root, if you are going to use Gradio, Press Enter...\n\n"
    fi
    launch_webui "$@"

}

start_initialization "$@"
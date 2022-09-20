#!/bin/bash -i

# Start the Stable Diffusion WebUI for Linux Users

DIRECTORY="."
ENV_NAME="ldm"
ENV_FILE="environment.yaml"
ENV_UPDATED=0
ENV_MODIFIED=$(date -r $ENV_FILE "+%s")
ENV_MODIFED_FILE=".env_updated"
if [[ -f $ENV_MODIFED_FILE ]]; then 
    ENV_MODIFIED_CACHED=$(<${ENV_MODIFED_FILE})
else 
    ENV_MODIFIED_CACHED=0
fi

conda_env_setup () {
    # Set conda path if it is not already in default environment
    CUSTOM_CONDA_PATH=

    # Allow setting custom path via file to allow updates of this script without undoing custom path
    if [ -f custom-conda-path.txt ]; then
        CUSTOM_CONDA_PATH=$(cat custom-conda-path.txt)
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

conda_env_activation () {
    # Activate conda environment
    conda activate $ENV_NAME
    conda info | grep active
}

sd_model_loading () {
    # Check to see if the SD model already exists, if not then it creates it and prompts the user to add the SD AI models to the Models directory
    if [ -f "$DIRECTORY/models/ldm/stable-diffusion-v1/model.ckpt" ]; then
        printf "AI Model already in place. Continuing...\n\n"
    else
        printf "\n\n########## MOVE MODEL FILE ##########\n\n"
        printf "Please download the 1.4 AI Model from Huggingface (or another source) and place it inside of the stable-diffusion-webui folder\n\n"
        read -p "Once you have sd-v1-4.ckpt in , Press Enter...\n\n"
        
        # Check to make sure checksum of models is the original one from HuggingFace and not a fake model set
        printf "fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556 sd-v1-4.ckpt" | sha256sum --check || exit 1
        mv sd-v1-4.ckpt $DIRECTORY/models/ldm/stable-diffusion-v1/model.ckpt
        rm -r ./Models
    fi
}

post_processor_model_loading () {
    # Check to see if GFPGAN has been added yet, if not it will download it and place it in the proper directory
    if [ -f "$DIRECTORY/src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth" ]; then
        printf "GFPGAN already exists. Continuing...\n\n"
    else
        printf "Downloading GFPGAN model. Please wait...\n"
        wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P $DIRECTORY/src/gfpgan/experiments/pretrained_models
    fi

    # Check to see if realESRGAN has been added yet, if not it will download it and place it in the proper directory
    if [ -f "$DIRECTORY/src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus.pth" ]; then
        printf "realESRGAN already exists. Continuing...\n\n"
    else
        printf "Downloading realESRGAN model. Please wait...\n"
        wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P $DIRECTORY/src/realesrgan/experiments/pretrained_models
        wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P $DIRECTORY/src/realesrgan/experiments/pretrained_models
    fi

    # Check to see if LDSR has been added yet, if not it will be cloned and its models downloaded to the correct directory
    if [ -f "$DIRECTORY/src/latent-diffusion/experiments/pretrained_models/model.ckpt" ]; then
        printf "LDSR already exists. Continuing...\n\n"
    else
        printf "Cloning LDSR and downloading model. Please wait...\n"
        git clone https://github.com/devilismyfriend/latent-diffusion.git
        mv latent-diffusion $DIRECTORY/src/latent-diffusion
        mkdir $DIRECTORY/src/latent-diffusion/experiments
        mkdir $DIRECTORY/src/latent-diffusion/experiments/pretrained_models
        wget https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1 -P $DIRECTORY/src/latent-diffusion/experiments/pretrained_models
        mv $DIRECTORY/src/latent-diffusion/experiments/pretrained_models/index.html?dl=1 $DIRECTORY/src/latent-diffusion/experiments/pretrained_models/project.yaml
        wget https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1 -P $DIRECTORY/src/latent-diffusion/experiments/pretrained_models
        mv $DIRECTORY/src/latent-diffusion/experiments/pretrained_models/index.html?dl=1 $DIRECTORY/src/latent-diffusion/experiments/pretrained_models/model.ckpt
    fi
}

launch_webui () {
    printf "\n\n########## LAUNCH USING GRADIO OR STREAMLIT? ##########\n\n"
    printf "Do you wish to run the Stable Diffusion WebUI using the Gradio or StreamLit Interface?\n\n"
    printf "Gradio: Currently Feature Complete, But Uses An Older Interface Style And Will Not Receive Major Updates\n"
    printf "StreamLit: Has A More Modern UI With More Features To Be Added And Will Be The Main UI Going Forward, But Currently In Active Development And Missing Some Gradio Features\n\n"
    printf "Which Version of the WebUI Interface do you wish to use?\n"
    select yn in "Gradio" "StreamLit"; do
        case $yn in
            Gradio ) printf "\nStarting Stable Diffusion WebUI: Gradio Interface. Please Wait...\n"; python scripts/relauncher.py; break;;
            StreamLit ) printf "\nStarting Stable Diffusion WebUI: StreamLit Interface. Please Wait...\n"; python -m streamlit run scripts/webui_streamlit.py; break;;
        esac
    done
}

start_initialization () {
    conda_env_setup
    sd_model_loading
    post_processor_model_loading
    conda_env_activation
    if [ ! -e "models/ldm/stable-diffusion-v1/model.ckpt" ]; then
        echo "Your model file does not exist! Place it in 'models/ldm/stable-diffusion-v1' with the name 'model.ckpt'."
        exit 1
    fi
    launch_webui

}

start_initialization

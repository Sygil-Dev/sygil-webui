#!/bin/bash
#
# Starts the gui using the conda env
#

ENV_NAME="ldm"
ENV_FILE="environment.yaml"
ENV_UPDATED=0
ENV_MODIFIED=$(date -r $ENV_FILE "+%s")
ENV_MODIFED_FILE=".env_updated"
if [[ -f $ENV_MODIFED_FILE ]]; then ENV_MODIFIED_CACHED=$(<${ENV_MODIFED_FILE}); else ENV_MODIFIED_CACHED=0; fi

# Set conda path if it is not already in default environment
custom_conda_path=

# Allow setting custom path via file to allow updates of this script without undoing custom path
if [ -f custom-conda-path.txt ]; then
    custom_conda_path=$(cat custom-conda-path.txt)
fi

# If custom path is set above, try to setup conda environment
if [ -f "${custom_conda_path}/etc/profile.d/conda.sh" ]; then
    . "${custom_conda_path}/etc/profile.d/conda.sh"
elif [ -n "${custom_conda_path}" ] && [ -f "${custom_conda_path}/bin" ]; then
    export PATH="${custom_conda_path}/bin:$PATH"
fi

if ! command -v conda >/dev/null; then
    echo "anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create/update conda env if needed
if ! conda env list | grep ".*${ENV_NAME}.*" >/dev/null 2>&1; then
    echo "Could not find conda env: ${ENV_NAME} ... creating ..."
    conda env create -f $ENV_FILE
    ENV_UPDATED=1
elif [[ ! -z $CONDA_FORCE_UPDATE && $CONDA_FORCE_UPDATE == "true" ]] || (( $ENV_MODIFIED > $ENV_MODIFIED_CACHED )); then
    echo "Updating conda env: ${ENV_NAME} ..."
    PIP_EXISTS_ACTION=w conda env update --file $ENV_FILE --prune
    ENV_UPDATED=1
fi

# Clear artifacts from conda after create/update
if (( $ENV_UPDATED > 0 )); then
    conda clean --all
    echo -n $ENV_MODIFIED > $ENV_MODIFED_FILE
fi

# Activate conda environment
conda activate $ENV_NAME
conda info | grep active

if [ ! -e "models/ldm/stable-diffusion-v1/model.ckpt" ]; then
    echo "Your model file does not exist! Place it in 'models/ldm/stable-diffusion-v1' with the name 'model.ckpt'."
    exit 1
fi

python scripts/relauncher.py

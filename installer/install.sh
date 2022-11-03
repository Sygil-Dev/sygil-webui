#!/bin/bash

# For developers only! Not for users!
# This script will install git and conda (if not found on the PATH variable)
#  using micromamba (an 8mb static-linked single-file binary, conda replacement).
# For users who already have git and conda, this step will be skipped.

# Then, it'll run the webui.cmd file to continue with the installation as usual.

# This enables a user to install this project without manually installing conda and git.

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Installing Sygil WebUI.."
echo ""

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="64";;
    arm64*)     OS_ARCH="aarch64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit
esac

# config
export MAMBA_ROOT_PREFIX="$(pwd)/installer_files/mamba"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MICROMAMBA_DOWNLOAD_URL="https://micro.mamba.pm/api/micromamba/linux-${OS_ARCH}/latest"
umamba_exists="F"

# figure out whether git and conda needs to be installed
if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

PACKAGES_TO_INSTALL=""

if ! hash "conda" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL conda"; fi
if ! hash "git" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"; fi

if "$MAMBA_ROOT_PREFIX/micromamba" --version &>/dev/null; then umamba_exists="T"; fi

# (if necessary) install git and conda into a contained environment
if [ "$PACKAGES_TO_INSTALL" != "" ]; then
    # download micromamba
    if [ "$umamba_exists" == "F" ]; then
        echo "Downloading micromamba from $MICROMAMBA_DOWNLOAD_URL to $MAMBA_ROOT_PREFIX/micromamba"

        mkdir -p "$MAMBA_ROOT_PREFIX"
        curl -L "$MICROMAMBA_DOWNLOAD_URL" | tar -xvj bin/micromamba -O > "$MAMBA_ROOT_PREFIX/micromamba"

        chmod u+x "$MAMBA_ROOT_PREFIX/micromamba"

        # test the mamba binary
        echo "Micromamba version:"
        "$MAMBA_ROOT_PREFIX/micromamba" --version
    fi

    # create the installer env
    if [ ! -e "$INSTALL_ENV_DIR" ]; then
        "$MAMBA_ROOT_PREFIX/micromamba" create -y --prefix "$INSTALL_ENV_DIR"
    fi

    echo "Packages to install:$PACKAGES_TO_INSTALL"

    "$MAMBA_ROOT_PREFIX/micromamba" install -y --prefix "$INSTALL_ENV_DIR" -c conda-forge $PACKAGES_TO_INSTALL

    if [ ! -e "$INSTALL_ENV_DIR" ]; then
        echo "There was a problem while initializing micromamba. Cannot continue."
        exit
    fi
fi

if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

conda activate

# run the installer script for linux
curl "https://raw.githubusercontent.com/JoshuaKimsey/Linux-StableDiffusion-Script/main/linux-sd.sh" > linux-sd.sh
chmod u+x linux-sd.sh

./linux-sd.sh

# tell the user that they need to download the ckpt
WEIGHTS_DOC_URL="https://sd-webui.github.io/stable-diffusion-webui/docs/2.linux-installation.html#initial-start-guide"

echo ""
echo "Now you need to install the weights for the stable diffusion model."
echo "Please follow the steps at $WEIGHTS_DOC_URL to complete the installation"

# it would be nice if the weights downloaded automatically, and didn't need the user to do this manually.

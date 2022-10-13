#!/bin/bash

# This script will install git and conda (if not found on the PATH variable)
#  using micromamba (an 8mb static-linked single-file binary, conda replacement).
# For users who already have git and conda, this step will be skipped.

# Then, it'll run the webui.cmd file to continue with the installation as usual.

# This enables a user to install this project without manually installing conda and git.

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x64";;
    arm64*)     OS_ARCH="arm64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit
esac

# config
export MAMBA_ROOT_PREFIX="$(pwd)/installer_files/mamba"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MICROMAMBA_BINARY_FILE="$(pwd)/installer_files/micromamba_linux_${OS_ARCH}"

# figure out whether git and conda needs to be installed
PACKAGES_TO_INSTALL=""

if ! hash "conda" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL conda"; fi
if ! hash "git" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"; fi
if ! hash "curl" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL curl"; fi

# (if necessary) install git and conda into a contained environment
if [ "$PACKAGES_TO_INSTALL" != "" ]; then
    # initialize micromamba
    if [ ! -e "$MAMBA_ROOT_PREFIX" ]; then
        mkdir -p "$MAMBA_ROOT_PREFIX"
        cp "$MICROMAMBA_BINARY_FILE" "$MAMBA_ROOT_PREFIX/micromamba"

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
fi

if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

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

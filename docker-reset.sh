#!/bin/bash
# Use this script to reset your Docker-based Stable Diffusion environment
# This script will remove all cached files/models that are downloaded during your first startup


declare -a deletion_paths=("src"
                            "gfpgan"
                            "sd_webui.egg-info"
                            ".env_updated"     # Check if still needed
                            )


# TODO This should be improved to be safer
install_dir=$(pwd)

echo $install_dir
read -p "Do you want to reset the above directory? (y/n) " -n 1 DIRCONFIRM
echo ""

if [[ $DIRCONFIRM =~ ^[Yy]$ ]]; then
    docker compose down
    docker image rm stable-diffusion-webui:dev
    docker volume rm stable-diffusion-webui_root_profile

    for path in "${deletion_paths[@]}"
    do
        echo "Removing files located at path: $install_dir/$path"
        rm -rf $path
    done
else
    echo "Exited without reset"
fi

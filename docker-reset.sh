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

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
import os
from omegaconf import OmegaConf
import urllib.request
import logging

logger = logging.getLogger(__name__)


def download_file(download_link, folder, file_name):
    model_path = os.path.join(folder, file_name)
    if os.path.exists(model_path):
        logger.info(file_name + " already exists")
        return
    os.makedirs(folder, exist_ok=True)
    logger.info('Downloading ' + file_name + '...')
    try:
        urllib.request.urlretrieve(download_link, model_path)
        logger.info("Downloaded " + file_name)
    except:
        logger.error("Failed to download " + file_name + " from " + download_link)
        raise


def update_models(models=None):
    if models is None:
        models = OmegaConf.load("configs/webui/webui_streamlit.yaml").model_manager.models

    for model_name, model in models.items():
        model_folder = model.save_location
        logger.info("Preparing model: " + model_name)
        for file_ in model.files.values():
            download_file(file_.download_link, model_folder, file_.file_name)
        logger.info("Model " + model_name + " is ready.")


if __name__ == "__main__":
    update_models()

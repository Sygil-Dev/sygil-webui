# webui_utils.py

# imports
import os, yaml
from PIL import Image
from pprint import pprint


# logging
log_file = 'webui_flet.log'

def log_message(message):
    with open(log_file,'a+') as log:
        log.write(message)


# Settings
path_to_default_config = 'configs/webui/webui_flet.yaml'
path_to_user_config = 'configs/webui/userconfig_flet.yaml'

def get_default_settings_from_config():
    with open(path_to_default_config) as f:
        default_settings = yaml.safe_load(f)
    return default_settings

def get_user_settings_from_config():
    settings = get_default_settings_from_config()
    if os.path.exists(path_to_user_config):
        with open(path_to_user_config) as f:
            user_settings = yaml.safe_load(f)
        settings.update(user_settings)
    return settings

def save_user_settings_to_config(settings):
    with open(path_to_user_config, 'w+') as f:
        yaml.dump(settings, f, default_flow_style=False)


# Image handling

def load_images(images):        # just for testing, needs love to function
    images_loaded = {}
    images_not_loaded = []
    for i in images:
        try:
            img = Image.open(images[i]['path'])
            if img:
                images_loaded.update({images[i].name:img})
        except:
            images_not_loaded.append(i)

    return images_loaded, images_not_loaded

def create_blank_image():
    img = Image.new('RGBA',(512,512),(0,0,0,0))
    return img


# Textual Inversion
textual_inversion_grid_row_list = [
	'model', 'medium', 'artist', 'trending', 'movement', 'flavors', 'techniques', 'tags',
	]

def run_textual_inversion(args):
    pass

# webui_utils.py

# imports
import os, yaml
from pprint import pprint

 
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


# Textual Inversion
textual_inversion_grid_row_list = [
	'model', 'medium', 'artist', 'trending', 'movement', 'flavors', 'techniques', 'tags',
	]


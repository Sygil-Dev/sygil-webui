# flet_utils.py

# imports
import os, yaml, js2py
from PIL import Image
from datetime import datetime
from pprint import pprint


# logging
def log_message(message):
    log_file = None
    # get time and format message
    prefix = datetime.now()
    msg_prefix = prefix.strftime("%Y/%m/%d %H:%M:%S")
    message = msg_prefix + message
    # check to see if we're appending to current logfile or making a new one'
    try:
        log_file = log_message.log
    except AttributeError:
        log_prefix = prefix.strftime("%Y%m%d_%H%M%S")
        log_message.log = os.path.join('logs/webui/',log_prefix+'webui_flet.log')
        log_file = log_message.log
    # write message to logfile
    with open(log_file,'a+') as log:
        log.write(message)

# settings
path_to_default_config = 'configs/webui/webui_flet.yaml'
path_to_user_config = 'configs/webui/userconfig_flet.yaml'

def get_default_settings_from_config():
    with open(path_to_default_config) as f:
        default_settings = yaml.safe_load(f)
    return default_settings

def get_user_settings_from_config():
    # get default settings
    settings = get_default_settings_from_config()
    # check to see if userconfig exists
    if os.path.exists(path_to_user_config):
        # compare to see which is newer
        default_time = os.path.getmtime(path_to_default_config)
        user_time = os.path.getmtime(path_to_user_config)
        # if default is newer, save over userconfig and exit early
        if (default_time > user_time):
            save_user_settings_to_config(settings)
            return settings
        # else, load userconfig
        with open(path_to_user_config) as f:
            user_settings = yaml.safe_load(f)
        settings.update(user_settings)
    # regardless, return settings
    return settings

def save_user_settings_to_config(settings):
    with open(path_to_user_config, 'w+') as f:
        yaml.dump(settings, f, default_flow_style=False)


# image handling
def create_blank_image():
    img = Image.new('RGBA',(512,512),(0,0,0,0))
    return img



# textual inversion
textual_inversion_grid_row_list = [
	'model', 'medium', 'artist', 'trending', 'movement', 'flavors', 'techniques', 'tags',
	]

def run_textual_inversion(args):
    pass

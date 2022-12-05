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
    message = msg_prefix + "  " + message
    # check to see if we're appending to current logfile or making a new one'
    try:
        log_file = log_message.log
    except AttributeError:
        log_prefix = prefix.strftime("%Y%m%d_%H%M%S")
        os.makedirs('log/webui/flet', exist_ok=True)
        log_message.log = os.path.join('log/webui/flet',log_prefix+'webui_flet.log')
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
    # regardless, return settings as dict
    return settings

def save_user_settings_to_config(settings):
    with open(path_to_user_config, 'w+') as f:
        yaml.dump(settings, f, default_flow_style=False)


# image handling
path_to_assets = "webui/flet/assets"
path_to_uploads = "webui/flet/uploads"

# creates blank image   (to do: take size as arg)
def create_blank_image():
    img = Image.new('RGBA',(512,512),(0,0,0,0))
    return img

# takes name of image
# returns dict
#   name of image : image handle
def get_image_from_uploads(name):
    path_to_image = os.path.join(path_to_uploads, name)
    if os.path.exists(path_to_image):
        img = Image.open(path_to_image)
        return {name:img}
    else:
        log_message(f'image not found: "{name}"')
        return {name:None}


# takes name of gallery as arg ('assets','output','uploads')
# returns list of dicts
#       name of image:
#           'img_path' : path_to_image
#           'info_path' : path_to_yaml
def get_gallery_images(gallery_name):
    path_to_gallery = None
    if gallery_name == 'uploads':
        path_to_gallery = path_to_uploads
    else:
        log_message(f'gallery not found: "{gallery_name}"')
        return None
    images = []
    files = os.listdir(path_to_gallery)
    for f in files:
        if f.endswith(('.jpg','.jpeg','.png')):
            path_to_file = os.path.join(path_to_gallery,f)
            images.append({f:{'img_path':path_to_file}})
        if f.endswith(('.yaml')):
            path_to_file = os.path.join(path_to_gallery,f)
            images.append({f:{'info_path':path_to_file}})
    #pprint(images)
    return images


# textual inversion
textual_inversion_grid_row_list = [
	'model', 'medium', 'artist', 'trending', 'movement', 'flavors', 'techniques', 'tags',
	]

def run_textual_inversion(args):
    pass

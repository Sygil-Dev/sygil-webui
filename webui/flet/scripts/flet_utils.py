# flet_utils.py

# imports
import os, yaml, base64
from PIL import Image
from io import BytesIO
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
path_to_uploads = "webui/flet/assets/uploads"
path_to_outputs = "webui/flet/assets/outputs"

# creates blank image   (to do: take size as arg)
def create_blank_image(size):
	try:
		create_blank_image.count +=1
	except AttributeError:
		create_blank_image.count = 1
	name = 'blank_layer_' + str(create_blank_image.count).zfill(2)
	img = Image.new('RGBA',size,(0,0,0,0))
	img.filename = name
	return img

# takes name of image
# returns dict
#   name of image : image handle
def get_image_from_uploads(name):
	path_to_image = os.path.join(path_to_uploads, name)
	if os.path.exists(path_to_image):
		image = Image.open(path_to_image)
		image = image.convert("RGBA")
		return image
	else:
		log_message(f'image not found: "{name}"')
		return None

def get_canvas_background(path):
	image = Image.open(path)
	image = image.convert("RGBA")
	return image

# takes list of Image(s) as arg
# returns single composite of all images
def get_visible_from_image_stack(image_list):
	visible_image = create_blank_image()
	for image in image_list:
		# need to crop images for composite
		x0, y0 = (image.width * 0.5) - 256, (image.height * 0.5) - 256
		x1, y1 = x0 + 512, y0 + 512
		box = (x0, y0, x1, y1)
		cropped_image = image.crop(box)
		visible_image = Image.alpha_composite(visible_image,cropped_image)
	return visible_image

# converts Image to base64 string
def convert_image_to_base64(image):
	image_buffer = BytesIO()
	image.save(image_buffer, format='PNG')
	image_buffer.seek(0)
	image_bytes = image_buffer.getvalue()
	return base64.b64encode(image_bytes).decode()

# takes name of gallery as arg ('assets','output','uploads')
# returns list of dicts
#       name of image:
#           'img_path' : path_to_image
#           'info_path' : path_to_yaml
def get_gallery_images(gallery_name):
	path_to_gallery = None
	images = []
	files = []
	if gallery_name == 'uploads':
		path_to_gallery = path_to_uploads
	elif gallery_name == 'outputs':
		path_to_gallery = path_to_outputs
	else:
		log_message(f'gallery not found: "{gallery_name}"')
		return images
	if os.path.exists(path_to_gallery):
		files = os.listdir(path_to_gallery)
	else:
		return None
	for f in files:
		if f.endswith(('.jpg','.jpeg','.png')):
			path_to_file = os.path.join('/uploads',f)
			images.append({f:{'img_path':path_to_file}})
		if f.endswith(('.yaml')):
			path_to_file = os.path.join('/uploads',f)
			images.append({f:{'info_path':path_to_file}})
	return images


# textual inversion
textual_inversion_grid_row_list = [
	'model', 'medium', 'artist', 'trending', 'movement', 'flavors', 'techniques', 'tags',
	]

def run_textual_inversion(args):
	pass

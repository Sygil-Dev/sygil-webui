# flet_utils.py

# imports
import os, yaml, base64
from PIL import Image, ImageDraw
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
	img.path = None
	return img

# takes name of image
# returns dict
#   name of image : image handle
def get_image_from_uploads(name):
	path_to_image = os.path.join(path_to_uploads, name)
	if os.path.exists(path_to_image):
		image = Image.open(path_to_image)
		image = image.convert("RGBA")
		image.filename = name
		image.path = path_to_image
		return image
	else:
		log_message(f'image not found: "{name}"')
		return None

def get_canvas_background(path):
	image = Image.open(path)
	image = image.convert("RGBA")
	return image

# make canvas frame
def get_canvas_frame(canvas_size):
	image = Image.new('RGBA',(4096,4096),(0,0,0,127))
	canvas_width = canvas_size[0]
	canvas_height = canvas_size[1]
	x0 = int((image.width - canvas_width) * 0.5)
	y0 = int((image.height - canvas_height) * 0.5)
	x1 = x0 + canvas_width
	y1 = y0 + canvas_height
	box = (x0, y0, x1, y1)
	image.paste((0,0,0,0), box)
	return convert_image_to_base64(image)

# takes list of Image(s) as arg
# returns single composite of all images
def get_preview_from_stack(size, stack):
	preview = Image.new('RGBA',size,(0,0,0,0))
	canvas_width = size[0]
	canvas_height = size[1]
	for layer in stack:
		image = layer.image
		# need to crop images for composite
		x0 = ((image.width - canvas_width) * 0.5) - layer.offset_x
		y0 = ((image.height - canvas_height) * 0.5) - layer.offset_y
		x1 = x0 + canvas_width
		y1 = y0 + canvas_height
		box = (x0, y0, x1, y1)
		cropped_image = image.crop(box)
		preview = Image.alpha_composite(preview,cropped_image)
	return preview

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
			image = Image.open(os.path.join(path_to_gallery,f))
			image = image.convert("RGBA")
			image.filename = f
			image.path = os.path.join(gallery_name,f)
			images.append(image)
	return images


# textual inversion
textual_inversion_grid_row_list = [
	'model', 'medium', 'artist', 'trending', 'movement', 'flavors', 'techniques', 'tags',
	]

def run_textual_inversion(args):
	pass

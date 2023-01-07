# Flet imports
import flet as ft

# other imports
import os
from math import pi
from typing import Optional
from loguru import logger
import logging
#logging.basicConfig(level=logging.DEBUG)

# utils imports
from scripts import flet_utils
from scripts.flet_settings_window import settings_window
from scripts.flet_gallery_window import gallery_window
from scripts.flet_file_manager import file_picker, uploads, imports
from scripts.flet_titlebar import titlebar
from scripts.flet_tool_manager import tool_manager
from scripts.flet_asset_manager import asset_manager, layer_action_menu
from scripts.flet_canvas import canvas
from scripts.flet_messages import messages
from scripts.flet_property_manager import property_manager

# for debugging
from pprint import pprint

os.environ["FLET_WS_MAX_MESSAGE_SIZE"] = "8000000"


#	main ###############################################################
@logger.catch(reraise=True)
def main(page: ft.Page):

#	init ###############################################################
	# messages
	page.messages = messages
	page.message = messages.message
	page.max_message_history = 50


	# ui
	page.current_layout = 'Default'
	page.titlebar_height = 50
	page.bottom_panel_height = page.height * 0.2
	page.toolbox_height = 250
	page.tool_manager_width = 50
	page.tool_manager_button_size = 40
	page.left_panel_width = 200
	page.right_panel_width = 250

	page.background_color = None
	page.primary_color = None
	page.secondary_color = 'white_10'
	page.tertiary_color = 'blue'

	page.text_color = None
	page.text_size = 14
	page.icon_size = 20

	page.padding = 0
	page.margin = 0
	page.container_padding = 0
	page.container_margin = 0

	page.tab_color = 'white_10'
	page.tab_padding = ft.padding.only(left = 2, top = 12, right = 2, bottom = 8)
	page.tab_margin = 0

	page.divider_height = 10
	page.vertical_divider_width = 10


	# titlebar
	page.titlebar = titlebar

	def change_theme_mode(e):
		page.theme_mode = "dark" if page.theme_mode == "light" else "light"

		if "(Light theme)" in titlebar.theme_switcher.tooltip:
			titlebar.theme_switcher.tooltip = titlebar.theme_switcher.tooltip.replace("(Light theme)", '')

		if "(Dark theme)" in titlebar.theme_switcher.tooltip:
			titlebar.theme_switcher.tooltip = titlebar.theme_switcher.tooltip.replace("(Dark theme)", '')

		titlebar.theme_switcher.tooltip += "(Light theme)" if page.theme_mode == "light" else "(Dark theme)"
		page.update()

	page.change_theme_mode = change_theme_mode


	# tools
	page.tool_manager = tool_manager
	page.current_tool = 'pan'

	def enable_tools():
		page.tool_manager.enable_tools()

	page.enable_tools = enable_tools

	def disable_tools():
		page.tool_manager.disable_tools()

	page.disable_tools = disable_tools

	def set_current_tool(e):
		page.tool_manager.clear_tools()
		page.canvas.clear_tools()
		e.control.selected = True
		page.current_tool = e.control.data['label']
		page.canvas.set_current_tool(e.control.data['label'])
		page.update()

	page.set_current_tool = set_current_tool


	# asset manager
	page.asset_manager = asset_manager
	page.active_layer = None
	page.visible_layers = []
	page.layer_height = 50

	def set_active_layer(layer_slot):
		if page.active_layer == layer_slot:
			return
		page.active_layer = layer_slot
		page.enable_tools()
		page.property_manager.refresh_layer_properties()

	page.set_active_layer = set_active_layer

	def add_blank_layer():
		image = flet_utils.create_blank_image(page.canvas_size)
		layer_slot = page.asset_manager.add_image_as_layer(image)
		layer_slot.layer_image = page.canvas.add_layer_image(image)
		page.message("added blank layer to canvas")
		page.refresh_layers()

	page.add_blank_layer = add_blank_layer

	def add_images_as_layers(images):
		layer_slots = page.asset_manager.add_images_as_layers(images)
		for slot in layer_slots:
			slot.layer_image = page.canvas.add_layer_image(slot.image)
			page.message(f'added "{slot.image.filename}" as layer')
		page.refresh_layers()

	page.add_images_as_layers = add_images_as_layers

	def load_images():
		page.file_picker.pick_files(file_type = 'image', allow_multiple = True)

	page.load_images = load_images


	# canvas
	page.canvas = canvas
	page.canvas_background = flet_utils.get_canvas_background('webui/flet/assets/images/default_grid_texture.png')
	page.canvas_size = [512,512]

	def get_viewport_size():
		viewport_width = page.width - (page.tool_manager_width + (page.vertical_divider_width * 3) + page.left_panel_width + page.right_panel_width)
		viewport_height = page.height - (page.titlebar_height * 2) - page.bottom_panel_height
		return viewport_width, viewport_height

	page.get_viewport_size = get_viewport_size


	def align_canvas():
		page.canvas.align_canvas()

	page.align_canvas = align_canvas


	# property manager
	page.property_manager = property_manager

	def refresh_canvas_preview():
		preview = page.canvas.get_image_stack_preview()
		page.property_manager.set_preview_image(preview)

	page.refresh_canvas_preview = refresh_canvas_preview

	def refresh_layers():
		if page.active_layer == None:
			page.disable_tools()
		else:
			page.enable_tools()
		page.asset_manager.refresh_layers()
		page.canvas.refresh_canvas()
		page.refresh_canvas_preview()
		page.property_manager.refresh_layer_properties()
		page.update()

	page.refresh_layers = refresh_layers


	# layouts
	def set_layout(e):
		page.current_layout = e.control.value
		page.update()

	page.set_layout = set_layout


	def on_page_change(e):
		page.titlebar.on_page_change()
		page.tool_manager.on_page_change()
		page.asset_manager.on_page_change()
		page.canvas.on_page_change()
		page.messages.on_page_change()
		page.property_manager.on_page_change()
		full_page.width = page.width
		full_page.height = page.height
		page.update()

	page.on_resize = on_page_change

	def on_window_change(e):
		if e.data == 'minimize' or e.data == 'close':
			return
		else:
			page.on_page_change(e)

	page.on_window_event = on_window_change

	# settings
	def load_settings():
		settings = flet_utils.get_user_settings_from_config()
		page.session.set('settings',settings)
		page.update()

	def save_settings_to_config():
		settings = page.session.get('settings')
		flet_utils.save_user_settings_to_config(settings)

	def reset_settings_from_config():
		settings = flet_utils.get_default_settings_from_config()
		page.session.remove('settings')
		page.session.set('settings',settings)
		save_settings_to_config()

	if not page.session.contains_key('settings'):
		load_settings()
		settings = page.session.get('settings')
		try:
			ui_settings = settings['webui_page']
			page.theme_mode = ui_settings['default_theme']['value']
			MAX_MESSAGE_HISTORY = ui_settings['max_message_history']['value']
		except AttributeError:
			page.message("Config load error: missing setting.",1)

		page.session.set('layout','default')


#	settings window ####################################################

	def close_settings_window(e):
		settings_window.open = False
		page.update()

	page.close_settings = close_settings_window

	def open_settings_window(e):
		page.dialog = settings_window
		settings_window.open = True
		page.update()

	page.open_settings = open_settings_window

	page.settings_window = settings_window
	settings_window.content.width = page.width * 0.50
	settings_window.content.bgcolor = page.primary_color
	settings_window.content.padding = page.container_padding
	settings_window.content.margin = page.container_margin


#	gallery window #####################################################

	def close_gallery_window(e):
		gallery_window.open = False
		page.update()

	page.close_gallery = close_gallery_window

	def open_gallery_window(e):
		page.dialog = gallery_window
		gallery_window.open = True
		page.update()

	page.open_gallery = open_gallery_window

	page.gallery_window = gallery_window
	page.refresh_gallery = gallery_window.refresh_gallery
	gallery_window.content.width = page.width * 0.5
	gallery_window.content.bgcolor = page.primary_color
	gallery_window.content.padding = page.container_padding
	gallery_window.content.margin = page.container_margin

	gallery_window.outputs_gallery.height = page.height * 0.75
	gallery_window.outputs_gallery.bgcolor = page.primary_color
	gallery_window.outputs_gallery.padding = page.container_padding
	gallery_window.outputs_gallery.margin = page.container_margin

	gallery_window.uploads_gallery.height = page.height * 0.75
	gallery_window.uploads_gallery.bgcolor = page.primary_color
	gallery_window.uploads_gallery.padding = page.container_padding
	gallery_window.uploads_gallery.margin = page.container_margin


#	file manager #######################################################

	def close_upload_window(e):
		uploads.open = False
		page.update()

	page.close_uploads = close_upload_window

	def open_upload_window(e):
		page.dialog = uploads
		uploads.open = True
		page.update()

	page.open_uploads = open_upload_window

	def close_import_window(e):
		imports.open = False
		page.update()

	page.close_imports = close_import_window

	def open_import_window(e):
		page.dialog = imports
		imports.open = True
		page.update()

	page.open_imports = open_import_window

	page.uploads = uploads
	page.imports = imports
	page.file_picker = file_picker
	page.overlay.append(file_picker)


#	center panel #############################################################

	text_editor = ft.Container(
			content = ft.Text('Under Construction.'),
			bgcolor = page.secondary_color,
			expand = True,
	)

	viewport = ft.Container(
			bgcolor = page.primary_color,
			padding = page.container_padding,
			margin = page.container_margin,
			content = ft.Tabs(
					selected_index = 0,
					animation_duration = 300,
					tabs = [
							ft.Tab(
									content = canvas,
									tab_content = ft.Text(
											value = 'Canvas',
											size = page.text_size,
									),
							),
							ft.Tab(
									text = 'Text Editor',
									content = text_editor,
									tab_content = ft.Text(
											value = 'Text Editor',
											size = page.text_size,
									),
							),
					],
			),
			expand = True,
	)

	center_panel = ft.Container(
			content = ft.Column(
					controls = [
							viewport,
							messages,
					],
			),
			bgcolor = page.primary_color,
			padding = page.container_padding,
			margin = page.container_margin,
			expand = True,
	)


#	layouts ############################################################

	default_layout = ft.Row(
			controls = [
				tool_manager,
				asset_manager,
				center_panel,
				property_manager,
			],
			expand=True,
	)

	current_layout = default_layout


#	workspace ##########################################################

	workspace = ft.Column(
			controls = [
				titlebar,
				current_layout,
			],
			expand = True,
	)

	page.workspace = workspace

	full_page = ft.Stack(
			expand = True,
			controls = [
				workspace,
				layer_action_menu,
			],
			height = page.height,
			width = page.width,
	)

	page.full_page = full_page

	page.title = "Stable Diffusion Playground"
	page.add(full_page)

	page.settings_window.setup(page.session.get('settings'))
	page.gallery_window.setup()
	page.titlebar.setup()
	page.tool_manager.setup()
	page.asset_manager.setup()
	page.canvas.setup()
	page.messages.setup()
	page.property_manager.setup()
	page.update()


ft.app(target=main, route_url_strategy="path", port=8505, assets_dir="assets", upload_dir="assets/uploads")

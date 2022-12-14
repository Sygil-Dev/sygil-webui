# Flet imports
import flet as ft

# other imports
from math import pi
from typing import Optional
from loguru import logger

# utils imports
from scripts import flet_utils
from scripts.flet_settings_window import settings_window
from scripts.flet_gallery_window import gallery_window
from scripts.flet_file_manager import file_picker, uploads, imports
from scripts.flet_appbar import appbar
from scripts.flet_tool_manager import toolbar
from scripts.flet_asset_manager import asset_manager
from scripts.flet_canvas import canvas
from scripts.flet_messages import messages
from scripts.flet_property_manager import property_manager

# for debugging
from pprint import pprint


#	main ###############################################################
@logger.catch(reraise=True)
def main(page: ft.Page):

#	init ###############################################################
	# messages
	page.message = messages.message
	page.max_message_history = 50

	# ui
	page.current_layout = 'Default'
	page.appbar_height = 50
	page.bottom_panel_height = page.height * 0.2
	page.toolbox_height = 250
	page.toolbar_width = 50
	page.toolbar_button_size = 40
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

	def change_theme_mode(e):
		page.theme_mode = "dark" if page.theme_mode == "light" else "light"

		if "(Light theme)" in appbar.theme_switcher.tooltip:
			appbar.theme_switcher.tooltip = appbar.theme_switcher.tooltip.replace("(Light theme)", '')

		if "(Dark theme)" in appbar.theme_switcher.tooltip:
			appbar.theme_switcher.tooltip = appbar.theme_switcher.tooltip.replace("(Dark theme)", '')

		appbar.theme_switcher.tooltip += "(Light theme)" if page.theme_mode == "light" else "(Dark theme)"
		page.update()

	page.change_theme_mode = change_theme_mode

	# layouts
	def set_layout(e):
		page.current_layout = e.control.value
		set_property_panel_options()
		page.update()

	page.set_layout = set_layout

	# tools
	page.current_tool = 'pan'

	# layer manager
	page.layer_list = []
	page.visible_layer_list = []
	page.active_layer_list = []

	# canvas
	page.canvas_background = flet_utils.get_canvas_background('webui/flet/assets/images/templategrid_albedo.png')
	page.canvas_size = (512,512)

	def get_viewport_size():
		viewport_width = page.width - (page.toolbar_width + (page.vertical_divider_width * 3) + page.left_panel_width + page.right_panel_width)
		viewport_height = page.height - (page.appbar_height * 3) - page.bottom_panel_height
		return (viewport_width, viewport_height)

	page.get_viewport_size = get_viewport_size

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
		if 'webui_page' in settings:
			if 'default_theme' in settings['webui_page']:
				page.theme_mode = settings['webui_page']['default_theme']['value']
			if 'max_message_history' in settings['webui_page']:
				MAX_MESSAGE_HISTORY = settings['webui_page']['max_message_history']['value']

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

	page.refresh_gallery = gallery_window.refresh_gallery

	page.gallery_window = gallery_window
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


#	file manager ######################################################

	def close_upload_window(e):
		uploads.open = False
		page.update()

	page.close_uploads = close_upload_window

	def open_upload_window(e):
		page.dialog = uploads
		uploads.content = page.selected_files
		uploads.open = True
		page.update()

	page.open_uploads = open_upload_window

	def close_import_window(e):
		imports.open = False
		page.update()

	page.close_imports = close_import_window

	def open_import_window(e):
		page.dialog = imports
		imports.content = page.selected_files
		imports.open = True
		page.update()

	page.open_imports = open_import_window

	page.selected_files = ft.Column(
			scroll = 'auto',
			tight = True,
			controls = [],
	);

	page.progress_bars: Dict[str, ft.ProgressBar] = {}
	page.uploads = uploads
	page.imports = imports
	page.file_picker = file_picker
	page.overlay.append(file_picker)


#	layouts ############################################################

	def set_current_tools():
		layout = page.current_layout
		if layout == 'Default':
			set_tools(default_tools)
		elif layout == 'Textual Inversion':
			set_tools(textual_inversion_tools)
		elif layout == 'Node Editor':
			set_tools(node_editor_tools)
		toolbar.update()

	def set_property_panel_options():
		layout = page.current_layout
		if layout == 'Default':
			set_properties(default_properties)
		elif layout == 'Textual Inversion':
			set_properties(textual_inversion_properties)
		elif layout == 'Node Editor':
			set_properties(node_editor_properties)


#	app bar ############################################################

	# have to rename appbar --> titlebar, because page.appbar is something we don't want.
	page.titlebar = appbar
	appbar.width = page.width
	appbar.height = page.appbar_height

	appbar.title.size = page.appbar_height * 0.5
	appbar.title.color = page.tertiary_color

	appbar.prompt.text_size = max(12,page.appbar_height * 0.25)
	appbar.prompt.focused_border_color = page.tertiary_color

	appbar.layout_menu.controls[0].text_size = page.text_size

	appbar.theme_switcher.size = page.appbar_height
	appbar.theme_switcher.icon_size = page.appbar_height * 0.5
	appbar.theme_switcher.tooltip = f"Click to change between the light and dark themes. Current {'(Light theme)' if page.theme_mode == 'light' else '(Dark theme)'}"
	appbar.theme_switcher.on_click = page.change_theme_mode

	appbar.settings_button.size = page.appbar_height
	appbar.settings_button.icon_size = page.appbar_height * 0.5
	appbar.settings_button.on_click = page.open_settings


#	toolbar ############################################################

	page.toolbar = toolbar
	toolbar.width = page.toolbar_width
	toolbar.bgcolor = page.primary_color
	toolbar.padding = page.container_padding
	toolbar.margin = page.container_margin

	toolbar.toolbox.bgcolor = page.secondary_color
	toolbar.toolbox.padding = page.container_padding
	toolbar.toolbox.margin = page.container_margin

	toolbar.tool_divider.content.height = page.divider_height
	toolbar.tool_divider.content.color = page.tertiary_color

	toolbar.tool_properties.bgcolor = page.secondary_color
	toolbar.tool_properties.padding = page.container_padding
	toolbar.tool_properties.margin = page.container_margin

	toolbar.dragbar.content.width = page.vertical_divider_width
	toolbar.dragbar.content.color = page.tertiary_color


#	layer manager ######################################################

	page.asset_manager = asset_manager
	asset_manager.width = page.left_panel_width
	asset_manager.bgcolor = page.primary_color
	asset_manager.padding = page.container_padding
	asset_manager.margin = page.container_margin
	asset_manager.set_tab_text_size(page.text_size)
	asset_manager.set_tab_bgcolor(page.secondary_color)
	asset_manager.set_tab_padding(page.container_padding)
	asset_manager.set_tab_margin(page.container_margin)

	asset_manager.dragbar.content.width = page.vertical_divider_width
	asset_manager.dragbar.content.color = page.tertiary_color


#	canvas #############################################################

	page.canvas = canvas
	canvas.bgcolor = page.secondary_color
	canvas.padding = page.container_padding
	canvas.margin = page.container_margin

	canvas.overlay.tools.zoom_in = page.icon_size
	canvas.overlay.tools.zoom_out = page.icon_size

	canvas.overlay.canvas_size.content.color = page.text_color
	canvas.overlay.canvas_size.content.size = page.text_size


#	text editor ########################################################

	text_editor = ft.Container(
			content = ft.Text('Under Construction.'),
			bgcolor = page.secondary_color,
			expand = True,
	)


#	viewport ##########################################################

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


#	bottom_panel #######################################################

	page.messages = messages
	messages.height = page.bottom_panel_height
	messages.bgcolor = page.primary_color
	messages.padding = page.container_padding
	messages.margin = page.container_margin
	messages.set_tab_text_size(page.text_size)
	messages.set_tab_bgcolor(page.secondary_color)
	messages.set_tab_padding(page.container_padding)
	messages.set_tab_margin(page.container_margin)

	messages.dragbar.content.height = page.divider_height
	messages.dragbar.content.color = page.tertiary_color


#	center panel #######################################################

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


#	property manager ###################################################

	page.property_manager = property_manager
	property_manager.width = page.right_panel_width
	property_manager.bgcolor = page.primary_color
	property_manager.padding = page.container_padding
	property_manager.margin = page.container_margin
	property_manager.set_tab_text_size(page.text_size)
	property_manager.set_tab_bgcolor(page.secondary_color)
	property_manager.set_tab_padding(page.container_padding)
	property_manager.set_tab_margin(page.container_margin)

	property_manager.dragbar.content.width = page.vertical_divider_width
	property_manager.dragbar.content.color = page.tertiary_color


#	layouts ############################################################

	default_layout = ft.Row(
			controls = [
				toolbar,
				asset_manager,
				center_panel,
				property_manager,
			],
			expand=True,
	)

	current_layout = default_layout


#	workspace ##########################################################

	workspace = ft.Container(
			bgcolor = page.background_color,
			padding = 0,
			margin = 0,
			expand = True,
			content = ft.Column(
					controls = [
						appbar,
						current_layout,
					],
			),
			height = page.height,
			width = page.width,
	)

	page.title = "Stable Diffusion Playground"
	page.add(workspace)

	page.settings_window.setup(page.session.get('settings'))
	page.gallery_window.setup()
	page.toolbar.setup()
	page.asset_manager.setup()
	page.canvas.setup()


ft.app(target=main, port= 8505, assets_dir="assets", upload_dir="assets/uploads")

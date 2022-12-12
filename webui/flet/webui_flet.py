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
from scripts.flet_tool_manager import toolbar
from scripts.flet_layer_manager import layer_manager
from scripts.flet_canvas import Canvas, ImageStack

# for debugging
from pprint import pprint


#	main ###############################################################
@logger.catch(reraise=True)
def main(page: ft.Page):

#	init ###############################################################
	# messages
	def message(text, err = 0):
		if err:
			text = "ERROR:  " + text
		add_message_to_messages(err,text)
		flet_utils.log_message(text)

	page.message = message

	# ui
	page.current_layout = 'Default'
	page.appbar_height = 50
	page.bottom_panel_height = page.height * 0.2
	page.toolbox_height = 250
	page.toolbar_width = 50
	page.toolbar_button_size = 40
	page.left_panel_width = 250
	page.right_panel_width = 250

	page.primary_color = None
	page.secondary_color = 'white_10'
	page.tertiary_color = 'blue'

	page.text_size = 20
	page.icon_size = 20
	page.container_padding = 0
	page.container_margin = 0
	page.tab_color = 'white_10'
	page.tab_padding = ft.padding.only(left = 2, top = 12, right = 2, bottom = 8)
	page.tab_margin = 0
	page.divider_height = 10
	page.vertical_divider_width = 10

	def change_theme_mode(e):
		page.theme_mode = "dark" if page.theme_mode == "light" else "light"

		if "(Light theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Light theme)", '')

		if "(Dark theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Dark theme)", '')

		theme_switcher.tooltip += "(Light theme)" if page.theme_mode == "light" else "(Dark theme)"
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
	page.canvas_size = [512,512]

	def get_viewport_size():
		viewport_width = page.width - (page.toolbar_width + (page.vertical_divider_width * 3) + page.left_panel_width + page.right_panel_width)
		viewport_height = page.height - (page.appbar_height * 3) - page.bottom_panel_height
		return (viewport_width, viewport_height)

	page.get_viewport_size = get_viewport_size

	def refresh_canvas():
		image_list = [page.canvas_background]
		for layer in page.visible_layer_list:
			image_list.append(layer.data['image'])
		image_stack.image_list = image_list
		image_stack.image = flet_utils.get_visible_from_image_stack(image_list)
		image_stack.image_base64 = flet_utils.convert_image_to_base64(image_stack.image)
		image_stack.content.controls[0].src_base64 = image_stack.image_base64
		image_stack.update()

	page.refresh_canvas = refresh_canvas

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

	page.gallery_window = gallery_window
	gallery_window.content.width = page.width * 0.5
	gallery_window.content.bgcolor = page.primary_color
	gallery_window.content.padding = page.container_padding
	gallery_window.content.margin = page.container_margin

	gallery_window.outputs_gallery.bgcolor = page.primary_color
	gallery_window.outputs_gallery.padding = page.container_padding
	gallery_window.outputs_gallery.margin = page.container_margin

	gallery_window.uploads_gallery.bgcolor = page.primary_color
	gallery_window.uploads_gallery.padding = page.container_padding
	gallery_window.uploads_gallery.margin = page.container_margin


#	upload window ######################################################
	def close_upload_window(e):
		upload_window.open = False
		page.update()

	def open_upload_window(e):
		page.dialog = upload_window
		upload_window.open = True
		page.update()

	def upload_file(e):
		if file_picker.result is not None and file_picker.result.files is not None:
			file_list = []
			for f in file_picker.result.files:
				upload_url = page.get_upload_url(f.name, 600)
				img = ft.FilePickerUploadFile(f.name,upload_url)
				file_list.append(img)
			file_picker.upload(file_list)

	def upload_complete(e):
		progress_bars.clear()
		selected_files.controls.clear()
		close_upload_window(e)
		page.message('File upload(s) complete.')
		layer_manager.add_images_as_layers(file_picker.images)
		file_picker.images.clear()
		refresh_gallery('uploads')

	def get_image_from_uploads(name):
		return flet_utils.get_image_from_uploads(name)

	def get_file_display(name, progress):
		display = ft.Column(
				controls = [
						ft.Row([ft.Text(name)]),
						progress,
				],
		)
		return display

	selected_files = ft.Column(
			scroll = 'auto',
			tight = True,
			controls = [],
	);
	progress_bars: Dict[str, ft.ProgressBar] = {}

	upload_window = ft.AlertDialog(
		title = ft.Text("Confirm file upload(s)"),
		content = selected_files,
		actions_alignment = "center",
		actions = [
			ft.ElevatedButton("UPLOAD", on_click = upload_file),
			ft.TextButton("CANCEL", on_click = close_upload_window),
		],
	)


#	import window ######################################################
	def close_import_window(e):
		import_window.open = False
		page.update()

	def open_import_window(e):
		page.dialog = import_window
		gallery_window.open = True
		page.update()

	def import_file(e):
		close_import_window(e)
		pass

	import_window = ft.AlertDialog(
		title=ft.Text("Confirm file import(s)"),
		content=selected_files,
		actions_alignment="center",
		actions=[
			ft.ElevatedButton("IMPORT", on_click = import_file),
			ft.TextButton("CANCEL", on_click = close_import_window),
		],
	)


#	file picker ########################################################
	def pick_images(e: ft.FilePickerResultEvent):
		progress_bars.clear()
		selected_files.controls.clear()
		# check to see if files or directory were chosen
		if e.files is not None and e.path is None:
			for f in e.files:
				prog = ft.ProgressBar(
						value = 0,
						color = 'blue',
				)
				progress_bars[f.name] = prog
				selected_files.controls.append(get_file_display(f.name,prog))
				file_picker.pending += 1
			# import if local, upload if remote
			if not e.page.web:
				open_import_window(e)
			else:
				open_upload_window(e)

	def on_image_upload(e: ft.FilePickerUploadEvent):
		if e.error:
			page.message(f"Upload error occurred! Failed to fetch '{e.file_name}'.",1)
			file_picker.pending -= 1
		else:
			# update progress bars
			progress_bars[e.file_name].value = e.progress
			progress_bars[e.file_name].update()
			if e.progress >= 1:
				file_picker.pending -= 1
				file_picker.images.update(get_image_from_uploads(e.file_name))
		if file_picker.pending <= 0:
			file_picker.pending = 0
			upload_complete(e)

	file_picker = ft.FilePicker(
			on_result = pick_images,
			on_upload = on_image_upload
	)

	page.file_picker = file_picker
	page.overlay.append(file_picker)
	file_picker.pending = 0
	file_picker.images = {}

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
	app_bar_title = ft.Text(
			value = "Sygil",
			size = page.appbar_height * 0.5,
			color = page.tertiary_color,
			text_align = 'center',
	)

	prompt = ft.TextField(
			value = "",
			min_lines = 1,
			max_lines = 1,
			content_padding = 10,
			shift_enter = True,
			tooltip = "Prompt to use for generation.",
			autofocus = True,
			hint_text = "A corgi wearing a top hat as an oil painting.",
			height = page.appbar_height,
	)

	generate_button = ft.ElevatedButton(
			text = "Generate",
			on_click = None,
			height = page.appbar_height,
	)

	layout_menu = ft.Row(
			alignment = 'start',
			controls = [
					ft.Dropdown(
							options = [
								ft.dropdown.Option(text="Default"),
								ft.dropdown.Option(text="Textual Inversion"),
								ft.dropdown.Option(text="Node Editor"),
							],
							value = 'Default',
							content_padding = 10,
							width = 200,
							on_change = page.set_layout,
							tooltip = "Switch between different workspaces",
							height = page.appbar_height,
					)
			],
			height = page.appbar_height,
	)

	theme_switcher = ft.IconButton(
			ft.icons.WB_SUNNY_OUTLINED,
			on_click = page.change_theme_mode,
			expand = 1,
			tooltip = f"Click to change between the light and dark themes. Current {'(Light theme)' if page.theme_mode == 'light' else '(Dark theme)'}",
			height = page.appbar_height,
			)

	settings_button = ft.IconButton(
			icon = ft.icons.SETTINGS,
			on_click = page.open_settings,
			height = page.appbar_height,
	)

	option_list = ft.Row(
			controls = [
				ft.Container(expand = 2, content = layout_menu),
				ft.Container(expand = 1, content = theme_switcher),
				ft.Container(expand = 1, content = settings_button),
			],
			height = page.appbar_height,
	)

	appbar = ft.Row(
			width = page.width,
			controls = [
					ft.Container(content = app_bar_title),
					ft.VerticalDivider(width = 20, opacity = 0),
					ft.Container(expand = 6, content = prompt),
					#ft.Container(expand = 1, content = generate_button),
					ft.Container(expand = 4, content = option_list),
			],
			height = page.appbar_height,
	)


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

	toolbar.toolbar_dragbar.content.width = page.vertical_divider_width
	toolbar.toolbar_dragbar.content.color = page.tertiary_color


#	layer manager ######################################################

	page.layer_manager = layer_manager
	layer_manager.bgcolor = page.secondary_color
	layer_manager.padding = page.container_padding
	layer_manager.margin = page.container_margin


#	asset manager ######################################################
	asset_manager = ft.Container(
			content = ft.Column(
					controls = [
							ft.Divider(height=10, opacity = 0),
							ft.Text("Under Construction"),
					],
			),
			bgcolor = page.secondary_color,
			padding = page.container_padding,
			margin = page.container_margin,
	)


#	left panel ###################################################
	def resize_left_panel(e: ft.DragUpdateEvent):
		page.left_panel_width = max(250, page.left_panel_width + e.delta_x)
		left_panel.width = page.left_panel_width
		page.update()

	left_panel_dragbar = ft.GestureDetector(
			mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
			drag_interval = 50,
			on_pan_update = resize_left_panel,
			content = ft.VerticalDivider(
					width = page.vertical_divider_width,
					color = page.tertiary_color,
			),
	)

	left_panel = ft.Container(
			width = page.left_panel_width,
			bgcolor = page.primary_color,
			padding = page.container_padding,
			margin = page.container_margin,
			clip_behavior = 'antiAlias',
			content = ft.Row(
					controls = [
						ft.Column(
							controls = [
								ft.Tabs(
										selected_index = 0,
										animation_duration = 300,
										tabs = [
											ft.Tab(
													text = "Layers",
													content = layer_manager,
											),
											ft.Tab(
													text = "Assets",
													content = asset_manager,
											),
										],
								),
							],
							alignment = 'start',
							expand = True
						),
						left_panel_dragbar,
					],
					expand = True,
			),
	)

#	canvas #############################################################

	def set_current_tool(tool):
		page.current_tool = tool
		if tool == 'pan':
			image_stack.mouse_cursor = ft.MouseCursor.GRAB
			image_stack.on_pan_update = pan_canvas
			image_stack.on_scroll = None
		elif tool == 'zoom':
			image_stack.mouse_cursor = ft.MouseCursor.MOVE
			image_stack.on_pan_update = None
			image_stack.on_scroll = zoom_canvas
		elif tool == 'move':
			image_stack.mouse_cursor = ft.MouseCursor.MOVE
			image_stack.on_pan_update = None
			image_stack.on_scroll = None
		elif tool == 'brush':
			image_stack.mouse_cursor = ft.MouseCursor.PRECISE
			image_stack.on_pan_update = None
			image_stack.on_scroll = None
		image_stack.update()

	def pan_canvas(e: ft.DragUpdateEvent):
		e.control.top = e.control.top + e.delta_y
		e.control.left = e.control.left + e.delta_x
		e.control.update()

	def zoom_canvas(e: ft.ScrollEvent):
		e.control.content.controls[0].width = e.control.content.controls[0].width - e.scroll_delta_y
		e.control.content.controls[0].height = e.control.content.controls[0].height - e.scroll_delta_y
		e.control.update()

	def move_layer(e: ft.DragUpdateEvent):
		pass

	def paint_layer(e: ft.DragUpdateEvent):
		pass

	# ImageStack == ft.GestureDetector
	image_stack = ImageStack(
			mouse_cursor = ft.MouseCursor.GRAB,
			drag_interval = 50,
			on_pan_update = pan_canvas,
			on_scroll = zoom_canvas,
			left = ((page.get_viewport_size())[0] * 0.5) - (page.canvas_size[0] * 0.5),
			top = ((page.get_viewport_size())[1] * 0.5) - (page.canvas_size[1] * 0.5),
			content = ft.Stack(
					[
						ft.Image(
								src_base64 = flet_utils.convert_image_to_base64(page.canvas_background),
								width = page.canvas_size[0],
								height = page.canvas_size[1],
								gapless_playback = True,
								expand = True,
						),
					],
			),
	)

	def zoom_in_canvas(e):
		pass

	def zoom_out_canvas(e):
		pass

	zoom_in_button = ft.IconButton(
			width = page.toolbar_button_size,
			icon_size = page.toolbar_button_size * 0.5,
			content = ft.Icon(ft.icons.ZOOM_IN_OUTLINED),
			tooltip = 'zoom in canvas',
			on_click = zoom_in_canvas,
	)

	zoom_out_button = ft.IconButton(
			width = page.toolbar_button_size,
			icon_size = page.toolbar_button_size * 0.5,
			content = ft.Icon(ft.icons.ZOOM_OUT_OUTLINED),
			tooltip = 'zoom out canvas',
			on_click = zoom_out_canvas,
	)

	canvas_tools = ft.Container(
			content = ft.Column(
					controls = [
						zoom_in_button,
						zoom_out_button,
					],
			),
			top = 4,
			right = 4,
	)

	canvas_overlay = ft.Stack(
			[
				canvas_tools,
				ft.Container(
						content = ft.Text(value = str(page.canvas_size)),
						bottom = 4,
						right = 4,
				)
			],
	)

	# Canvas == ft.Container
	canvas = Canvas(
			content = ft.Stack(
					[
						image_stack,
						canvas_overlay,
					],
					clip_behavior = None,
			),
			alignment = ft.alignment.center,
			expand = True,
			bgcolor = page.secondary_color,
			padding = page.container_padding,
			margin = page.container_margin,
	)

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
									text = 'Canvas',
									content = canvas,
							),
							ft.Tab(
									text = 'Text Editor',
									content = text_editor,
							),
					],
			),
			expand = True,
	)

#	bottom_panel #######################################################

	def prune_messages():
		if len(messages.controls) > MAX_MESSAGE_HISTORY:
			messages.controls.pop(0)
		messages.update()

	def add_message_to_messages(err,text):
		if err:
			msg = ft.Text(value = text, color = ft.colors.RED)
		else:
			msg = ft.Text(value = text)
		messages.controls.append(msg)
		prune_messages()

	messages = ft.Column(
			spacing = 0,
			scroll = 'auto',
			auto_scroll = True,
			controls = [],
	)
	messages_window = ft.Container(
			bgcolor = page.secondary_color,
			padding = page.container_padding,
			margin = page.container_margin,
			content = messages,
	)
	video_editor_window = ft.Column(
			expand = True,
			controls = [ft.Text("Under Construction")]
	)

	bottom_panel = ft.Container(
			height = page.bottom_panel_height,
			bgcolor = page.primary_color,
			padding = page.container_padding,
			margin = page.container_margin,
			content = ft.Tabs(
					selected_index = 0,
					animation_duration = 300,
					tabs = [
							ft.Tab(
									text = "Messages",
									content = messages_window,
							),
							ft.Tab(
									text = "Video Editor",
									content = video_editor_window,
							),
					],
			)
	)
#	center panel #######################################################

	center_panel = ft.Container(
			content = ft.Column(
					controls = [
							viewport,
							bottom_panel,
					],
			),
			bgcolor = page.primary_color,
			padding = page.container_padding,
			margin = page.container_margin,
			expand = True,
	)


#	properties #########################################################
	# canvas layout properties
	model_menu = ft.Dropdown(
			label = "Custom Models",
			options = [
				ft.dropdown.Option("Stable Diffusion 1.5"),
				ft.dropdown.Option("Waifu Diffusion 1.3"),
				ft.dropdown.Option("MM-27 Merged Pruned"),
			],
			height = 70,
			expand = 1,
			content_padding = 0,
			value = "Stable Diffusion 1.5",
			tooltip = "Custom models located in your `models/custom` folder including the default stable diffusion model.",
	)

	sampling_menu = ft.Dropdown(
			label = "Sampling method",
			options = [ #["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
				ft.dropdown.Option("k_lms"),
				ft.dropdown.Option("k_euler"),
				ft.dropdown.Option("k_euler_a"),
				ft.dropdown.Option("k_dpm_2"),
				ft.dropdown.Option("k_dpm_2_a"),
				ft.dropdown.Option("k_heun"),
				ft.dropdown.Option("PLMS"),
				ft.dropdown.Option("DDIM"),
			],
			height = 70,
			expand = 1,
			content_padding = 10,
			value = "k_lms",
			tooltip = "Sampling method or scheduler to use, different sampling method"
					" or schedulers behave differently giving better or worst performance in more or less steps."
					"Try to find the best one for your needs and hardware.",
	)

	default_properties = ft.Container(
			bgcolor = page.tab_color,
			padding = page.tab_padding,
			margin = page.tab_margin,
			content = ft.Column(
					spacing = 12,
					controls = [
							ft.Row(
								controls = [
									model_menu,
								],
							),
							ft.Row(
								controls = [
									sampling_menu,
								],
							),
							ft.Row(
								controls = [
									ft.TextField(label="Width", value=512, height=50, expand=1, content_padding=10, suffix_text="W", text_align='center', tooltip="Widgth in pixels.", keyboard_type="number"),
									ft.TextField(label="Height", value=512, height=50, expand=1, content_padding=10, suffix_text="H", text_align='center', tooltip="Height in pixels.",keyboard_type="number"),
								],
							),
							ft.Row(
								controls = [
									ft.TextField(label="CFG", value=7.5, height=50, expand=1, content_padding=10, text_align='center', #suffix_text="CFG",
										tooltip="Classifier Free Guidance Scale.", keyboard_type="number"),
									ft.TextField(label="Sampling Steps", value=30, height=50, expand=1, content_padding=10, text_align='center', tooltip="Sampling steps.", keyboard_type="number"),
								],
							),
							ft.Row(
								controls = [
									ft.TextField(
											label = "Seed",
											hint_text = "blank=random seed",
											height = 50,
											expand = 1,
											text_align = 'start',
											content_padding = 10,
											#suffix_text = "seed",
											tooltip = "Seed used for the generation, leave empty or use -1 for a random seed. You can also use word as seeds.",
											keyboard_type = "number"
									),
								],
							),
							ft.Divider(height = 10, color = page.tertiary_color),
					]
			),
			expand = True
	)


	# textual inversion layout properties
	def set_clip_model(e):
		pass

	clip_model_menu = ft.Dropdown(
			label = "Clip Model",
			value = 0,
			options = [
				ft.dropdown.Option(key = 0, text="Vit-L/14"),
				ft.dropdown.Option(key = 1, text="Vit-H-14"),
				ft.dropdown.Option(key = 2, text="Vit-g-14"),
			],
			tooltip = "Select Clip model to use.",
			on_change = set_clip_model,
	)

	other_model_menu_label = ft.Text(value='Other Models', tooltip = "For DiscoDiffusion and JAX enable all the same models here as you intend to use when generating your images.")
	other_model_menu = ft.PopupMenuButton(
			items = [
				ft.PopupMenuItem(text="VitL14_336px", checked=False, data='VitL14_336px', on_click=None),
				ft.PopupMenuItem(text="VitB16", checked=False, data='VitB16', on_click=None),
				ft.PopupMenuItem(text="VitB32", checked=False, data='VitB32', on_click=None),
				ft.PopupMenuItem(text="RN50", checked=False, data='RN50', on_click=None),
				ft.PopupMenuItem(text="RN50x4", checked=False, data='RN50x4', on_click=None),
				ft.PopupMenuItem(text="RN50x16", checked=False, data='RN50x16', on_click=None),
				ft.PopupMenuItem(text="RN50x64", checked=False, data='RN50x64', on_click=None),
				ft.PopupMenuItem(text="RN101", checked=False, data='RN101', on_click=None),
			],
	)

	def get_textual_inversion_settings():
		settings = {
			'selected_models' : [],
			'selected_images' : [],
			'results' : [],
		}
		return settings

	def get_textual_inversion_grid_row(row_name):
		row_items = []
		row_items.append(ft.Text(value = row_name))
		row_items.append(ft.Text(value = flet_utils.get_textual_inversion_row_value(row_name)))
		return row_items

	def get_textual_inversion_results_grid():
		grid_rows = []
		for item in flet_utils.textual_inversion_grid_row_list:
			grid_rows.append(
				ft.Row(
					controls = get_textual_inversion_grid_row(item),
					height = 50,
				)
			)
		return ft.Column(controls = grid_rows)

	def get_textual_inversion_results(e):
		e.control.data = get_textual_inversion_settings()
		flet_utils.run_textual_inversion(e.control.data)
		textual_inversion_results.content = get_textual_inversion_results_grid()
		page.update()

	run_textual_inversion_button =  ft.ElevatedButton("Get Text from Image(s)", on_click=get_textual_inversion_results, data = {})

	textual_inversion_results = ft.Container(content = None)

	textual_inversion_properties = ft.Container(
			bgcolor = page.tab_color,
			padding = page.tab_padding,
			margin = page.tab_margin,
			content = ft.Column(
					controls = [
							ft.Row(
								controls = [
									clip_model_menu,
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Row(
								controls = [
									other_model_menu_label,
									other_model_menu,
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Row(
								controls = [
									run_textual_inversion_button,
								],
								alignment = 'spaceAround',
							),
							ft.Divider(height=10, color = page.tertiary_color),
							ft.Row(
								controls = [
									textual_inversion_results,
								],
								wrap = True,
							)
					]
			),
			expand = True
	)


	# node editor layout properties
	node_editor_properties = ft.Container(
			bgcolor = page.tab_color,
			padding = page.tab_padding,
			margin = page.tab_margin,
			content = ft.Column(
					controls = [
							ft.Text("Under Construction")
					]
			),
			expand = True
	)

	current_properties = default_properties

	def set_properties(control):
		property_panel.content.controls[0] = control
		property_panel.update()

#	property panel #####################################################
	property_panel = ft.Container(
			bgcolor = page.tab_color,
			padding = page.tab_padding,
			margin = page.tab_margin,
			content = ft.Column(
					spacing = 0,
					controls = [
							current_properties,
					],
			),
	)

# 	advanced panel #####################################################
	advanced_panel = ft.Container(
			bgcolor = page.tab_color,
			padding = page.tab_padding,
			margin = page.tab_margin,
			content = ft.Column(
					controls = [
							ft.Text("Under Construction."),
					],
			),
	)

#	right panel ########################################################
	def resize_right_panel(e: ft.DragUpdateEvent):
		page.right_panel_width = max(250, page.right_panel_width - e.delta_x)
		right_panel.width = page.right_panel_width
		page.update()

	right_panel_dragbar = ft.GestureDetector(
			mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
			drag_interval = 50,
			on_pan_update = resize_right_panel,
			content = ft.VerticalDivider(
					width = page.vertical_divider_width,
					color = page.tertiary_color,
			),
	)

	right_panel = ft.Container(
			width = page.right_panel_width,
			bgcolor = page.primary_color,
			padding = page.container_padding,
			margin = page.container_margin,
			content = ft.Row(
					controls = [
						right_panel_dragbar,
						ft.Column(
								controls = [
									ft.Tabs(
											selected_index = 0,
											animation_duration = 300,
											tabs = [
													ft.Tab(
															text = 'Properties',
															content = property_panel,
													),
													ft.Tab(
															text = 'Advanced',
															content = advanced_panel,
													),
											],
									),
								],
								alignment = 'start',
								expand = True
						),
					],
					expand = True,
			),
	)


#	layouts ############################################################

	default_layout = ft.Row(
			controls = [
				toolbar,
				left_panel,
				center_panel,
				right_panel,
			],
			expand=True,
	)

	current_layout = default_layout

#	workspace ##########################################################
	def draggable_out_of_bounds(e):
		if e.data == 'false':
			if layer_manager.data['layer_being_moved']:
				index = layer_manager.data['layer_being_moved'].data['index']
				layer_manager.insert_layer_slot(index)

	catchall = ft.DragTarget(
			group = 'catchall',
			on_will_accept = draggable_out_of_bounds,
			content = ft.Container(
				width = page.width,
				height = page.height,
			),
	)

	workspace = ft.Column(
			controls = [
					appbar,
					current_layout,
			],
	)


#	make page ##########################################################
	full_page = ft.Container(
			bgcolor = page.primary_color,
			padding = 0,
			margin = 0,
			expand = True,
			content = ft.Stack(
					[
							catchall,
							workspace,
					],
					height = page.height,
					width = page.width,
			),
	)

	page.title = "Stable Diffusion Playground"
	page.add(full_page)

	page.settings_window.setup(page.session.get('settings'))
	page.gallery_window.setup()
	page.toolbar.setup()
	page.layer_manager.update_layers()


ft.app(target=main, port= 8505, assets_dir="assets", upload_dir="assets/uploads")

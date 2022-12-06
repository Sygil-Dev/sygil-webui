# Flet imports
import flet as ft
from flet.ref import Ref

# other imports
from math import pi
from typing import Optional
from loguru import logger

# utils imports
from scripts import flet_utils
from scripts.flet_layer_manager import LayerManager
from scripts.flet_settings_window import SettingsWindow

# for debugging
from pprint import pprint

# custom classes

class GalleryWindow(ft.AlertDialog):
	def empty(self):
		pass

#	main ###############################################################
@logger.catch(reraise=True)
def main(page: ft.Page):

	def message(text, err = 0):
		if err:
			text = "ERROR:  " + text
		add_message_to_messages(err,text)
		flet_utils.log_message(text)

	page.message = message

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

#	init ###############################################################
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

	def open_settings_window(e):
		page.dialog = settings_window
		settings_window.open = True
		page.update()

	def apply_settings(e):
		settings_window.update_settings_window()

	def save_settings(e):
		save_settings_to_config()
		settings_window.update_settings_window()

	def reset_settings(e):
		reset_settings_from_config()
		settings_window.update_settings_window()

	settings_window = SettingsWindow(
			title = ft.Text("Settings"),
			content = ft.Container(
					width = page.width * 0.50,
					content = ft.Tabs(
							selected_index = 0,
							animation_duration = 300,
							tabs = None,
					),
			),
			actions = [
					ft.ElevatedButton(
							text = "Apply",
							icon = ft.icons.CHECK_CIRCLE,
							on_click = apply_settings,
					),
					ft.ElevatedButton(
							text = "Save",
							icon = ft.icons.SAVE,
							on_click = save_settings,
					),
					ft.ElevatedButton(
							text = "Restore Defaults",
							icon = ft.icons.RESTORE_FROM_TRASH_ROUNDED,
							on_click = reset_settings,
					),
			],
			actions_alignment = "end",
	)

	settings_window.get_settings_window_tabs(page.session.get('settings'))


#	gallery window #####################################################
	def close_gallery_window(e):
		gallery_window.open = False
		page.update()

	def open_gallery_window(e):
		page.dialog = gallery_window
		gallery_window.open = True
		page.update()

	def get_gallery_images(gallery_name):
		return flet_utils.get_gallery_images(gallery_name)

	def refresh_gallery(gallery_name):
		index = None
		if gallery_name == 'uploads':
			index = 0
		elif gallery_name == 'outputs':
			index = 1
		else:
			page.message(f'{gallery_name} gallery not found.', 1)
			return None
		gallery_window.content.content.tabs[index].content = get_gallery_display(gallery_name)

	def get_gallery_display(gallery_name):
		gallery_display = ft.Stack(
				[
						ft.Row(
								controls = None,
								wrap = False,
								scroll = 'always',
								expand = True,
						),
						ft.Column(
								controls = [
										ft.Row(
												controls = [
														ft.IconButton(
																height = page.height * 0.75,
																icon_size = 50,
																content = ft.Icon(ft.icons.ARROW_CIRCLE_LEFT_OUTLINED),
																tooltip = 'last image',
																on_click = None,
														),
														ft.IconButton(
																height = page.height * 0.75,
																icon_size = 50,
																content = ft.Icon(ft.icons.ARROW_CIRCLE_RIGHT_OUTLINED),
																tooltip = 'next image',
																on_click = None,
														),
												],
												expand = True,
												alignment = 'spaceBetween',
										),
								],
								alignment = 'center',
						),
				],
		)
		gallery = get_gallery_images(gallery_name)
		if len(gallery) < 1:
			gallery_display.controls[0].controls.append(
					ft.Image(
							src = '/images/chickens.jpg',
							tooltip = 'Nothing here but us chickens!',
							gapless_playback = True,
					)
			)
			return gallery_display
			
		for i in range(len(gallery)):
			image = gallery[i]
			image_name = list(image.keys())[0]
			image_path = image[image_name]['img_path']
			image_data = None
			if 'info_path' in image[image_name]:
				image_data = image[image_name]['info_path']
			gallery_display.controls[0].controls.append(
					ft.Image(
							src = image_path,
							tooltip = image_name,
							gapless_playback = True,
					)
			)
		return gallery_display


	gallery_window = GalleryWindow(
			title = ft.Text('Gallery'),
			content = ft.Container(
					width = page.width * 0.5,
					content = ft.Tabs(
							selected_index = 0,
							animation_duration = 300,
							tabs = [
								ft.Tab(
										text = "Uploads",
										content = get_gallery_display('uploads'),
								),
								ft.Tab(
										text = "Outputs",
										content = get_gallery_display('outputs'),
								),
							],
					),
			),
			actions = [
					ft.ElevatedButton(
							text = "Save",
							icon = ft.icons.SAVE,
							on_click = None,
					),
					ft.ElevatedButton(
							text = "Discard",
							icon = ft.icons.RESTORE_FROM_TRASH_ROUNDED,
							on_click = None,
					),
			],
			actions_alignment="end",
	)


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
		#modal = True,
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
		#modal=True,
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
	file_picker.pending = 0
	file_picker.images = {}
	page.overlay.append(file_picker)

#	layouts ############################################################
	def change_layout(e):
		page.session.set('layout',e.control.value)
		#set_current_options()
		set_current_tools()
		set_property_panel_options()
		page.update()

#	def set_current_layout_options():
		#layout = page.session.get('layout')
		#if layout == 'Default':
		#	current_layout_options.controls.append(default_layout_options)
		#elif layout == 'Textual Inversion':
		#	current_layout_options.controls.append(textual_inversion_layout_options)
		#elif layout == 'Node Editor':
		#	current_layout_options.controls.append(node_editor_layout_options)

	def set_current_tools():
		layout = page.session.get('layout')
		if layout == 'Default':
			set_tools(default_tools)
		elif layout == 'Textual Inversion':
			set_tools(textual_inversion_tools)
		elif layout == 'Node Editor':
			set_tools(node_editor_tools)
		toolbar.update()

	def set_property_panel_options():
		layout = page.session.get('layout')
		if layout == 'Default':
			set_properties(default_properties)
		elif layout == 'Textual Inversion':
			set_properties(textual_inversion_properties)
		elif layout == 'Node Editor':
			set_properties(node_editor_properties)


#	app bar ############################################################
	app_bar_title = ft.Text(
			value = "Sygil",
			size = 20,
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
			height = 50,
	)

	generate_button = ft.ElevatedButton(
			text = "Generate",
			on_click=None,
			height = 50,
	)


#	default_layout_options = ft.Row(
			#alignment = 'start',
			#controls = [
			#	ft.Container(ft.IconButton(content = ft.Text(value = 'Canvas'), tooltip ='Canvas Options', on_click = None, disabled=True)),
			#	ft.Container(ft.IconButton(content = ft.Text(value = 'Layers'), tooltip ='Layer Options', on_click = None, disabled=True)),
			#	ft.Container(ft.IconButton(content = ft.Text(value = 'Tools'), tooltip ='Toolbox', on_click = None, disabled=True)),
			#	ft.Container(ft.IconButton(content = ft.Text(value = 'Preferences'), tooltip ='Set Editor Preferences', on_click = None, disabled=True)),
			#],
			#height = 50,
#	)

#	textual_inversion_layout_options = ft.Row(
			#alignment = 'start',
			#controls = [
			#	ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='textual_inversion options 1', on_click = None, disabled=True)),
			#	ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 2', on_click = None, disabled=True)),
			#	ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 3', on_click = None, disabled=True)),
			#],
			#height = 50,
#	)

#	node_editor_layout_options = ft.Row(
			#alignment = 'start',
			#controls = [
			#	ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='node_editor options 1', on_click = None, disabled=True)),
			#	ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 2', on_click = None, disabled=True)),
			#	ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 3', on_click = None, disabled=True)),
			#],
			#height = 50,
#	)

#	current_layout_options = ft.Row(
			#alignment = 'start',
			#controls = [
			#	ft.Container(content = default_layout_options),
			#],
			#height = 50,
#	)

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
							on_change = change_layout,
							tooltip = "Switch between different workspaces",
							height = 50,
					)
			],
			height = 50,
	)

	def change_theme(e):
		page.theme_mode = "dark" if page.theme_mode == "light" else "light"

		if "(Light theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Light theme)", '')

		if "(Dark theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Dark theme)", '')

		theme_switcher.tooltip += "(Light theme)" if page.theme_mode == "light" else "(Dark theme)"
		page.update()

	theme_switcher = ft.IconButton(
			ft.icons.WB_SUNNY_OUTLINED,
			on_click = change_theme,
			expand = 1,
			tooltip = f"Click to change between the light and dark themes. Current {'(Light theme)' if page.theme_mode == 'light' else '(Dark theme)'}",
			height = 50,
			)

	settings_button = ft.IconButton(
			icon = ft.icons.SETTINGS,
			on_click = open_settings_window,
			height = 50,
	)

	option_list = ft.Row(
			controls = [
				#ft.Container(expand=True, content = current_layout_options),
				ft.Container(expand = 2, content = layout_menu),
				ft.Container(expand = 1, content = theme_switcher),
				ft.Container(expand = 1, content = settings_button),
				#ft.Container(expand = 1, content = menu_button),
			],
			height = 50,
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
			height = 50,
	)


#	toolbar ############################################################
	def add_blank_layer(e):
		layer_manager.add_blank_layer(e)

	open_gallery_button = ft.IconButton(
			width = 50,
			content = ft.Icon(ft.icons.DASHBOARD_OUTLINED),
			tooltip = 'Gallery',
			on_click = open_gallery_window,
	)

	add_blank_layer_button = ft.IconButton(
			width = 50,
			content = ft.Icon(ft.icons.ADD_OUTLINED),
			tooltip = 'add new blank layer',
			on_click = add_blank_layer,
	)

	add_image_as_layer_button = ft.IconButton(
			width = 50,
			content = ft.Icon(ft.icons.IMAGE_OUTLINED),
			tooltip = 'load image(s) as new layer(s)',
			on_click = lambda _: file_picker.pick_files(file_type = 'image', allow_multiple = True),
	)

	toolbox = ft.Container(
			content = ft.Row(
					alignment = 'start',
					wrap = True,
					controls = [
						open_gallery_button,
						add_blank_layer_button,
						add_image_as_layer_button,
					]
			)
	)

	tool_properties = ft.Container(
			content = ft.Column(
					controls = [],
			)
	)

	toolbar = ft.Container(
			width = 50,
			content = ft.Column(
					controls = [
						toolbox,
						tool_properties,
					],
			),
	)

#	layer manager ######################################################
	layer_manager = LayerManager(
			content = None,
			padding = ft.padding.only(top = 4),
			bgcolor = ft.colors.WHITE10,
			data = {
				'layer_list': [],
				'active_layer_list': [],
				'layer_being_moved': None,
				'layer_last_index': -1,
			},
	)

	layer_manager.content = layer_manager.make_layer_holder()


#	asset manager ######################################################
	asset_manager = ft.Container(
			content = ft.Column(
					controls = [
							ft.Divider(height=10, opacity = 0),
							ft.Text("Under Construction"),
					],
			),
			bgcolor = ft.colors.WHITE10,
	)


#	layers/asset tab ###################################################
	layers = ft.Container(
			width = 200,
			content = ft.Tabs(
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
	)

#	canvas #############################################################
	canvas = ft.Container(
			content = ft.Stack(
					[
						ft.Image(
								src=f"https://i.redd.it/qdxksbar05o31.jpg",
								#width=300,
								#height=300,
								#fit="contain",
								gapless_playback=True,
								expand=True,
						),
					],
					clip_behavior = None,
			),
			alignment = ft.alignment.center,
			expand = True,
	)


#	text editor ########################################################
	text_editor = ft.Container(
			content = ft.Text('Under Construction.'),
			expand = True,
	)

#	top panel ##########################################################
	top_panel = ft.Container(
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
		page.update()

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
			bgcolor = ft.colors.BLACK12,
			content = messages,
	)
	video_editor_window = ft.Column(
			expand = True,
			controls = [ft.Text("Under Construction")]
	)

	bottom_panel = ft.Container(
			height = page.height * .2,
			padding = ft.padding.only(bottom = 12),
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
							top_panel,
							bottom_panel,
					],
			),
			expand = True,
	)


#	property panel #####################################################
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
							ft.Draggable(content=ft.Divider(height=10, color="gray")),
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
							ft.Draggable(content=ft.Divider(height=10, color="gray")),
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

	# property panel
	property_panel = ft.Container(
			padding = ft.padding.only(top = 12, left = 4, right = 4),
			bgcolor = ft.colors.WHITE10,
			content = ft.Column(
					spacing = 0,
					controls = [
							current_properties,
					],
			),
	)

# 	advanced panel #####################################################
	advanced_panel = ft.Container(
			bgcolor = ft.colors.WHITE10,
			content = ft.Column(
					controls = [
							ft.Text("Under Construction."),
					],
			),
	)

	right_panel = ft.Container(
			content = ft.Tabs(
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
			width = 250,
	)


#	layouts ############################################################

	default_layout = ft.Row(
			controls = [
				toolbar,
				ft.VerticalDivider(width=2, color="gray", opacity = 0),
				layers,
				ft.VerticalDivider(width=2, color="gray", opacity = 0),
				center_panel,
				ft.VerticalDivider(width=2, color="gray", opacity = 0),
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
	full_page = ft.Stack(
			[
					catchall,
					workspace,
			],
			height = page.height,
			width = page.width,
	)

	page.title = "Stable Diffusion Playground"
	page.add(full_page)


	layer_manager.update_layers()

ft.app(target=main, port= 8505, assets_dir="assets", upload_dir="assets/uploads")

# Flet imports
import flet as ft
from flet.ref import Ref

# other imports
from math import pi
from typing import Optional
from loguru import logger
# utils imports
import webui_flet_utils

# for debugging
from pprint import pprint

@logger.catch(reraise=True)
def main(page: ft.Page):
	## main function defines
	default_control_height = 50
	default_text_size_a = 15
	default_text_size_b = 20
	default_text_size_c = 25
	toolbar_width = 25

	def get_settings_from_config():
		pass

	def change_theme(e):
		page.theme_mode = "dark" if page.theme_mode == "light" else "light"

		if "(Light theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Light theme)", '')

		if "(Dark theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Dark theme)", '')

		theme_switcher.tooltip += "(Light theme)" if page.theme_mode == "light" else "(Dark theme)"
		page.update()

###### layouts #########################################################
	def change_layout(e):
		current_layout.value = e.control.data
		set_current_layout_options(e.control.data)
		set_current_layout_tools(e.control.data)
		set_property_panel_options(e.control.data)
		page.update()

	def set_current_layout_options(layout):
		for control in current_layout_options.controls:
			 current_layout_options.controls.pop()
		if layout == 'Latent Space':
			current_layout_options.controls.append(latent_space_layout_options)
		elif layout == 'Textual Inversion':
			current_layout_options.controls.append(textual_inversion_layout_options)
		elif layout == 'Node Editor':
			current_layout_options.controls.append(node_editor_layout_options)

	def set_current_layout_tools(layout):
		for control in current_layout_tools.controls:
			 current_layout_tools.controls.pop()
		if layout == 'Latent Space':
			current_layout_tools.controls.append(latent_space_layout_tools)
		elif layout == 'Textual Inversion':
			current_layout_tools.controls.append(textual_inversion_layout_tools)
		elif layout == 'Node Editor':
			current_layout_tools.controls.append(node_editor_layout_tools)

	def set_property_panel_options(layout):
		controls = property_panel.content.controls
		for control in controls:
			 controls.pop()
		if layout == 'Latent Space':
			controls.append(latent_space_layout_properties)
		elif layout == 'Textual Inversion':
			controls.append(textual_inversion_layout_properties)
		elif layout == 'Node Editor':
			controls.append(node_editor_layout_properties)


###### settings window #################################################
	def close_settings_window(e):
		settings.open = False
		page.update()

	def open_settings_window(e):
		page.dialog = settings
		settings.open = True
		page.update()

	general_settings = ft.Column(
			alignment = 'start',
			scroll = 'auto',
			controls = [
					ft.Divider(height=10, color="gray"),
					ft.Row(
						controls = [
								ft.Dropdown(
										label = "GPU",
										options = [
											ft.dropdown.Option("0:NVIDEA-BLAHBLAH9000"),
										],
										value = "0:NVIDEA-BLAHBLAH9000",
										tooltip = "Select which GPU to use.",
								),
						],
					),
					ft.Row(
						controls = [
								ft.TextField(
										label = "Output Directory",
										value = 'outputs',
										text_align = 'start',
										tooltip = "Choose output directory.",
										keyboard_type = 'text',
								),
						],
					),
					ft.Row(
						controls = [
								ft.Dropdown(
										label = "Default Model",
										options = [
											ft.dropdown.Option("Stable Diffusion v1.5"),
										],
										value = "",
										tooltip = "Select default model to use.",
								),
						],
					),
					ft.Row(
						controls = [
								ft.TextField(
										label = "Default Model Config",
										value = 'configs/stable-diffusion/v1-inference.yaml',
										text_align = 'start',
										tooltip = "Choose default model config.",
										keyboard_type = 'text',
								),
						],
					),
					ft.Row(
						controls = [
								ft.TextField(
										label = "Default Model Path",
										value = 'models/ldm/stable-diffusion-v1/Stable Diffusion v1.5.ckpt',
										text_align = 'start',
										tooltip = "Choose default model path.",
										keyboard_type = 'text',
								),
						],
					),
					ft.Row(
						controls = [
								ft.TextField(
										label = "Default GFPGAN Directory",
										value = 'models/gfpgan',
										text_align = 'start',
										tooltip = "Choose default gfpgan directory.",
										keyboard_type = 'text',
								),
						],
					),
					ft.Row(
						controls = [
								ft.TextField(
										label = "Default RealESRGAN Directory",
										value = 'models/gfpgan',
										text_align = 'start',
										tooltip = "Choose default realESRGAN directory.",
										keyboard_type = 'text',
								),
						],
					),
					ft.Row(
						controls = [
								ft.Dropdown(
										label = "Default RealESRGAN Model",
										options = [
											ft.dropdown.Option(""),
										],
										value = "",
										tooltip = "Select which realESRGAN model to use.",
								),
						],
					),
					ft.Row(
						controls = [
								ft.Dropdown(
										label = "Default Upscaler",
										options = [
											ft.dropdown.Option(""),
										],
										value = "",
										tooltip = "Select which upscaler to use.",
								),
						],
					),
			],
	)

	performance_settings = ft.Column(
			alignment = 'start',
			scroll = 'auto',
			controls = [
					ft.Divider(height=10, color="gray"),
			],
	)

	server_settings = ft.Column(
			alignment = 'start',
			scroll = True,
			controls = [
					ft.Divider(height=10, color="gray"),
			],
	)

	interface_settings = ft.Column(
			alignment = 'start',
			scroll = True,
			controls = [
					ft.Divider(height=10, color="gray"),
			],
	)

	settings = ft.AlertDialog(
			#modal = True,
			title = ft.Text("Settings"),
			content = ft.Container(
					width = page.width * 0.50,
					content = ft.Tabs(
							selected_index = 0,
							animation_duration = 300,
							tabs = [
								ft.Tab(
										text = "General",
										content = general_settings,
								),
								ft.Tab(
										text = "Performance",
										content = performance_settings,
								),
								ft.Tab(
										text = "Server",
										content = server_settings,
								),
								ft.Tab(
										text = "Interface",
										content = interface_settings,
								),
							],
					),
			),
			actions = [
					# should save options when clicked
					ft.ElevatedButton("Save", icon=ft.icons.SAVE, on_click=close_settings_window),
					# Should allow you to discard changed made to the settings.
					ft.ElevatedButton("Discard", icon=ft.icons.RESTORE_FROM_TRASH_ROUNDED, on_click=close_settings_window),
			],
			actions_alignment="end",
			#on_dismiss=lambda e: print("Modal dialog dismissed!"),
	)

###### gallery window ##################################################
	def close_gallery_window(e):
		gallery.open = False
		page.update()

	def open_gallery_window(e):
		page.dialog = gallery
		gallery.open = True
		page.update()

	gallery = ft.AlertDialog(
			title = ft.Text('Gallery'),
			content = ft.Row(
					controls = [
							ft.Text('Working on it I swear...'),
							ft.Container(
									width = page.width * 0.75,
									height = page.height * 0.75,
							),
					],
			),
			actions = [
					# should save image to disk
					ft.ElevatedButton("Save", icon=ft.icons.SAVE, on_click=close_settings_window),
					# remove image from gallery
					ft.ElevatedButton("Discard", icon=ft.icons.RESTORE_FROM_TRASH_ROUNDED, on_click=close_settings_window),
			],
			actions_alignment="end",
	)

###### app bar #########################################################
	app_bar_title = ft.Text(
			value = "Sygil",
			size = default_text_size_c,
			text_align = 'center',
	)

	prompt = ft.TextField(
			value = "",
			min_lines = 1,
			max_lines = 1,
			shift_enter = True,
			tooltip = "Prompt to use for generation.",
			autofocus = True,
			hint_text = "A corgi wearing a top hat as an oil painting.",
			height = default_control_height,
			text_size = default_text_size_b,
	)

	generate_button = ft.ElevatedButton(
			text = "Generate",
			on_click=None,
			height = default_control_height,
	)

	layouts = ft.PopupMenuButton(
			items = [
				ft.PopupMenuItem(text="Latent Space", on_click=change_layout, data="Latent Space"),
				ft.PopupMenuItem(text="Textual Inversion", on_click=change_layout, data="Textual Inversion"),
				ft.PopupMenuItem(text="Node Editor", on_click=change_layout, data="Node Editor"),
			],
			tooltip = "Switch between different workspaces",
			height = default_control_height,
	)

	current_layout = ft.Text(
			value = 'Latent Space',
			size = default_text_size_a,
			tooltip = "Current Workspace",
	)

	layout_menu = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(content = layouts),
				ft.Container(content = current_layout),
			],
			height = default_control_height,
	)

	latent_space_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Text(value = 'Canvas'), tooltip ='Canvas Options', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Layers'), tooltip ='Layer Options', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Tools'), tooltip ='Toolbox', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Preferences'), tooltip ='Set Editor Preferences', on_click = None, disabled=True)),
			],
			height = default_control_height,
	)

	textual_inversion_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='textual_inversion options 1', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 2', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 3', on_click = None, disabled=True)),
			],
			height = default_control_height,
	)

	node_editor_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='node_editor options 1', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 2', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 3', on_click = None, disabled=True)),
			],
			height = default_control_height,
	)

	current_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(content = latent_space_layout_options),
			],
			height = default_control_height,
	)

	theme_switcher = ft.IconButton(
			ft.icons.WB_SUNNY_OUTLINED,
			on_click = change_theme,
			expand = 1,
			tooltip = f"Click to change between the light and dark themes. Current {'(Light theme)' if page.theme_mode == 'light' else '(Dark theme)'}",
			height = default_control_height,
			)

	settings_button = ft.IconButton(
			icon = ft.icons.SETTINGS,
			on_click = open_settings_window,
			height = default_control_height,
	)

	menu_button = ft.PopupMenuButton(
			items = [
					#ft.PopupMenuItem(text="Settings", on_click=open_settings_modal),
					ft.PopupMenuItem(),  # divider
					#ft.PopupMenuItem(text="Checked item", checked=False, on_click=check_item_clicked),
			],
			height = default_control_height,
	)

	option_bar = ft.Row(
			controls = [
#                ft.Container(expand=True, content = current_layout_options),
				ft.Container(expand = 2, content = layout_menu),
				ft.Container(expand = 1, content = theme_switcher),
				ft.Container(expand = 1, content = settings_button),
				ft.Container(expand = 1, content = menu_button),
			],
			height = default_control_height,
	)

	appbar = ft.Row(
			width = page.width,
			controls = [
					ft.Container(content = app_bar_title),
					ft.VerticalDivider(width = 20, opacity = 0),
					ft.Container(expand = 6, content = prompt),
#					ft.Container(expand = 1, content = generate_button),
					ft.Container(expand = 4, content = option_bar),
			],
			height = default_control_height,
	)


###### toolbar #########################################################
	open_gallery_button = ft.IconButton(width = 20, content = ft.Icon(ft.icons.DASHBOARD_OUTLINED), tooltip = 'Gallery', on_click = open_gallery_window)
	import_image_button = ft.IconButton(width = 20, content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'Import image as new layer', on_click = None)

	universal_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
				open_gallery_button,
				import_image_button,
			]
	)

	## canvas layout tools
	latent_space_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	## textual inversion tools
	textual_inversion_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	## node editor tools
	node_editor_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	current_layout_tools = ft.Column(
			width = toolbar_width,
			controls = [
				latent_space_layout_tools,
			],
	)

	toolbar = ft.Column(
			width = toolbar_width,
			controls = [
				ft.Container(content = universal_tools),
				ft.Container(content = current_layout_tools),
			],
	)


###### layers panel ####################################################
	def show_hide_layer(e):
		parent = e.control.data['parent']
		if parent.data['hidden']:
			parent.data['hidden'] = False
			parent.opacity = 1.0
		else:
			parent.data['hidden'] = True
			parent.opacity = 0.5
		page.update()

	def get_layers():
		layers = [ft.Divider(height=10, opacity = 0)]
		count = 0
		for i in range(10):
			count += 1
			layer_icon = ft.IconButton(
					icon = ft.icons.HIGHLIGHT_ALT_OUTLINED,
					tooltip = 'show/hide',
					on_click = show_hide_layer,
					data = {'parent':None},
			)
			layer_label = ft.Text(value = ("layer_" + str(count)))
			layer_button = ft.Row(
					controls = [
							layer_icon,
							layer_label,
					],
					data = {'hidden':False},
			)
			layer_icon.data.update({'parent':layer_button})  ## <--see what i did there? :)
			layers.append(layer_button)
		return layers

	layer_list = get_layers()
	
	layer_manager = ft.Container(
			content = ft.Column(
					controls = layer_list,
			),
			bgcolor = ft.colors.WHITE10,
	)

	asset_manager = ft.Container(
			content = ft.Column(
					controls = [
							ft.Divider(height=10, opacity = 0),
					],
			),
			bgcolor = ft.colors.WHITE10,
	)

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

###### canvas ##########################################################
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


###### text editor #####################################################
	text_editor = ft.Container(
			content = ft.Text('WIP'),
			expand = True,
	)

###### top panel #######################################################
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

###### bottom_panel ####################################################
	video_editor_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)
	console_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)

	bottom_panel = ft.Row(
			height = 150,
			controls = [
				ft.Tabs(
						selected_index = 0,
						animation_duration = 300,
						tabs = [
								ft.Tab(
										text = "Video Editor",
										content = video_editor_window,
								),
								ft.Tab(
										text = "Console",
										content = console_window,
								),
						],
				),
			],
	)

###### center panel ####################################################

	center_panel = ft.Container(
			content = ft.Column(
					controls = [
							top_panel,
							bottom_panel,
					],
			),
			expand = True,
	)


###### property panel ##################################################
	## canvas layout properties
	model_menu = ft.Dropdown(
			label = "Custom Models",
			options = [
				ft.dropdown.Option("Stable Diffusion 1.5"),
				ft.dropdown.Option("Waifu Diffusion 1.3"),
				ft.dropdown.Option("MM-27 Merged Pruned"),
			],
			height = 70,
			expand = 1,
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
			value = "k_lms",
			tooltip = "Sampling method or scheduler to use, different sampling method"
					" or schedulers behave differently giving better or worst performance in more or less steps."
					"Try to find the best one for your needs and hardware.",
	)

	latent_space_layout_properties = ft.Container(
			content = ft.Column(
					controls = [
							ft.Row(
								controls = [
									model_menu,
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Row(
								controls = [
									sampling_menu,
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Row(
								controls = [
									ft.TextField(label="Width", value=512, height=50, expand=1, suffix_text="W", text_align='center', tooltip="Widgth in pixels.", keyboard_type="number"),
									ft.TextField(label="Height", value=512, height=50, expand=1, suffix_text="H", text_align='center', tooltip="Height in pixels.",keyboard_type="number"),
									ft.TextField(label="CFG", value=7.5, height=50, expand=1, text_align='center', #suffix_text="CFG",
										tooltip="Classifier Free Guidance Scale.", keyboard_type="number"),
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Row(
								controls = [
									ft.TextField(label="Seed", hint_text="blank=random seed", height=60, expand=2, text_align='start', #suffix_text="seed",
										tooltip="Seed used for the generation, leave empty or use -1 for a random seed. You can also use word as seeds.",
										keyboard_type="number"
									),
									ft.TextField(label="Sampling Steps", value=30, height=60, expand=1, text_align='center', tooltip="Sampling steps.", keyboard_type="number"),
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Draggable(content=ft.Divider(height=10, color="gray")),
#							ft.Switch(label="Stable Horde", value=False, disabled=True, tooltip="Option disabled for now."),
#							ft.Draggable(content=ft.Divider(height=10, color="gray")),
#							ft.Switch(label="Batch Options", value=False, disabled=True, tooltip="Option disabled for now."),
#							ft.Draggable(content=ft.Divider(height=10, color="gray")),
#							ft.Switch(label="Upscaling", value=False, disabled=True, tooltip="Option disabled for now."),
#							ft.Draggable(content=ft.Divider(height=10, color="gray")),
#							ft.Switch(label="Preview Image Settings", value=False, disabled=True, tooltip="Option disabled for now."),
#							ft.Draggable(content=ft.Divider(height=10, color="gray")),
					]
			),
			expand = True
	)


	## textual inversion layout properties
	clip_model_menu_label = ft.Text(value='Clip Models', tooltip = "Select Clip model(s) to use.")
	clip_model_menu = ft.PopupMenuButton(
			items = [
				ft.PopupMenuItem(text="Vit-L/14", checked=False, data='Vit-L/14', on_click=None),
				ft.PopupMenuItem(text="Vit-H-14", checked=False, data='Vit-H-14', on_click=None),
				ft.PopupMenuItem(text="Vit-g-14", checked=False, data='Vit-g-14', on_click=None),
			],
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
		row_items.append(ft.Text(value = webui_flet_utils.get_textual_inversion_row_value(row_name)))
		return row_items

	def get_textual_inversion_results_grid():
		grid_rows = []
		for item in webui_flet_utils.textual_inversion_grid_row_list:
			grid_rows.append(
				ft.Row(
					controls = get_textual_inversion_grid_row(item),
					height = 50,
				)
			)
		return ft.Column(controls = grid_rows)

	def get_textual_inversion_results(e):
		e.control.data = get_textual_inversion_settings()
		webui_flet_utils.run_textual_inversion(e.control.data)
		textual_inversion_results.content = get_textual_inversion_results_grid()
		page.update()

	run_textual_inversion_button =  ft.ElevatedButton("Get Text from Image(s)", on_click=get_textual_inversion_results, data = {})

	textual_inversion_results = ft.Container(content = None)

	textual_inversion_layout_properties = ft.Container(
			content = ft.Column(
					controls = [
							ft.Row(
								controls = [
									clip_model_menu_label,
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


	## node editor layout properties
	node_editor_layout_properties = ft.Container(
			content = ft.Column(
					controls = [
					]
			),
			expand = True
	)

	## property panel
	property_panel = ft.Container(
			bgcolor = ft.colors.WHITE10,
			content = ft.Column(
					controls = [
							ft.Divider(height=10, opacity = 0),
							latent_space_layout_properties,
					],
			),
	)
	
	## advanced panel
	advanced_panel = ft.Container(
			bgcolor = ft.colors.WHITE10,
			content = ft.Column(
					controls = [
							ft.Divider(height=10, opacity = 0),
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
			width = 400,
	)

###### workspace #######################################################
	workspace = ft.Row(
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


###### make page #######################################################
	page.title = "Stable Diffusion Playground"
	page.theme_mode = "dark"
	page.appbar = ft.AppBar(
			#leading=leading,
			#leading_width=leading_width,
			automatically_imply_leading=True,
			#elevation=5,
			bgcolor=ft.colors.BLACK26,
			actions=[appbar]
	)
	page.add(workspace)



ft.app(target=main, port=8505, view=ft.WEB_BROWSER)

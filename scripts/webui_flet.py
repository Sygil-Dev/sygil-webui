# Flet imports
import flet as ft
from flet.ref import Ref

# other imports
from math import pi
from typing import Optional
from loguru import logger
# utils imports
import webui_utils

# for debugging
from pprint import pprint

@logger.catch(reraise=True)
def main(page: ft.Page):
	## main function defines
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

###### layouts ###################################################################
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
		for control in property_panel.controls:
			 property_panel.controls.pop()
		if layout == 'Latent Space':
			property_panel.controls.append(latent_space_layout_properties)
		elif layout == 'Textual Inversion':
			property_panel.controls.append(textual_inversion_layout_properties)
		elif layout == 'Node Editor':
			property_panel.controls.append(node_editor_layout_properties)


###### orphans (for now) #######################################################
	prompt = ft.TextField(
			#label="Prompt",
			value="",
			min_lines=1,
			max_lines=1,
			shift_enter=True,
			#width=1000,
			tooltip="Prompt to use for generation.",
			#autofocus=True,
			hint_text="A corgi wearing a top hat as an oil paiting.",
	)

	generate_button = ft.ElevatedButton("Generate", on_click=None)


###### settings window #####################################################################
	def close_settings_window(e):
		settings.open = False
		page.update()

	def open_settings_window(e):
		page.dialog = settings
		settings.open = True
		page.update()

	general_settings = ft.Column(
			alignment = 'start',
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
			controls = [
					ft.Divider(height=10, color="gray"),
			],
	)

	server_settings = ft.Column(
			alignment = 'start',
			controls = [
					ft.Divider(height=10, color="gray"),
			],
	)

	interface_settings = ft.Column(
			alignment = 'start',
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

###### gallery window ###################################################################
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

###### app bar ###################################################################
	app_bar_title = ft.Text("Sygil", size = 25, text_align = 'center')

	layouts = ft.PopupMenuButton(
			items = [
				ft.PopupMenuItem(text="Latent Space", on_click=change_layout, data="Latent Space"),
				ft.PopupMenuItem(text="Textual Inversion", on_click=change_layout, data="Textual Inversion"),
				ft.PopupMenuItem(text="Node Editor", on_click=change_layout, data="Node Editor"),
			],
			tooltip = "Switch between different workspaces",
	)

	current_layout = ft.Text('Latent Space', size = 20, tooltip="Current Workspace")

	layout_menu = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(content = layouts),
				ft.Container(content = current_layout),
			]
	)

	latent_space_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Text(value = 'Canvas'), tooltip ='Canvas Options', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Layers'), tooltip ='Layer Options', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Tools'), tooltip ='Toolbox', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Preferences'), tooltip ='Set Editor Preferences', on_click = None, disabled=True)),
			]
	)

	textual_inversion_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='textual_inversion options 1', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 2', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 3', on_click = None, disabled=True)),
			]
	)

	node_editor_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='node_editor options 1', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 2', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 3', on_click = None, disabled=True)),
			]
	)

	current_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(content = latent_space_layout_options),
			]
	)

	theme_switcher = ft.IconButton(
			ft.icons.WB_SUNNY_OUTLINED,
			on_click = change_theme,
			expand = 1,
			tooltip = f"Click to change between the light and dark themes. Current {'(Light theme)' if page.theme_mode == 'light' else '(Dark theme)'}"
			)

	settings_button = ft.IconButton(icon=ft.icons.SETTINGS, on_click=open_settings_window)

	menu_button = ft.PopupMenuButton(
			items = [
					#ft.PopupMenuItem(text="Settings", on_click=open_settings_modal),
					ft.PopupMenuItem(),  # divider
					#ft.PopupMenuItem(text="Checked item", checked=False, on_click=check_item_clicked),
			]
	)

	option_bar = ft.Row(
			controls = [
				#ft.Container(expand = 1, content = dropdown),
				ft.Container(expand = 1, content = theme_switcher),
				ft.Container(expand = 1, content = settings_button),
				ft.Container(expand = 1, content = menu_button),
			]
	)

	appbar = ft.Row(
			width = page.width,
			controls = [
					ft.Container(width = 60, content = app_bar_title),
					ft.VerticalDivider(width = 10, color = "gray"),
					ft.Container(content = layout_menu),
					ft.VerticalDivider(width = 20, opacity = 0),
					ft.Container(expand=True, content = current_layout_options),
#					ft.Container(expand = 4, content = prompt),
#					ft.Container(expand = 1, content = generate_button),
					ft.VerticalDivider(width = 10, color = "gray"),
					ft.Container(width = 410, content = option_bar),
			],
	)


###### toolbar ########################################################
	open_gallery_button = ft.IconButton(width = 50, content = ft.Icon(ft.icons.DASHBOARD_OUTLINED), tooltip = 'Gallery', on_click = open_gallery_window)
	import_image_button = ft.IconButton(width = 50, content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'Import Image', on_click = None)
	universal_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
				open_gallery_button,
				import_image_button,
			]
	)

	## canvas layout tools
	def show_hide_layer(e):
		if e.control.data['hidden']:
			e.control.data['hidden'] = False
			e.control.opacity = 1.0
		else:
			e.control.data['hidden'] = True
			e.control.opacity = 0.5
		page.update()

	def get_layers():
		layers = []
		count = 0
		for i in range(10):
			count += 1
			label = "layer_" + str(count)
			layer_button = ft.IconButton(width = 50, content = ft.Icon(ft.icons.HIGHLIGHT_ALT_OUTLINED), tooltip = label, on_click = show_hide_layer, data = {'hidden':False})
			layers.append(layer_button)
		return layers

	layer_list = get_layers()

	latent_space_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = layer_list,
	)

	## textual inversion tools
	textual_inversion_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = layer_list,
	)

	## node editor tools
	node_editor_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	current_layout_tools = ft.Column(
			width = 50,
			controls = [
				latent_space_layout_tools,
			],
	)

	toolbar = ft.Column(
			width = 50,
			controls = [
				ft.Container(width = 50, content = universal_tools),
				ft.Divider(height = 10, color = "gray"),
				ft.Container(width = 50, content = current_layout_tools),
			],
	)


###### canvas #######################################################################
	canvas = ft.Container(
			content = ft.Stack(
					[
						#ft.Row([
						#ft.Image(
						#src=f"https://static.wixstatic.com/media/ac4dba_13be94c39c804e8aa2131a51036a0244~mv2.png",
						#width=300,
						#height=30,
						#fit="contain",
						#expand=True,
						#),
						#]),
						ft.Stack(
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
						),
					],
					#width=40,
					#height=40,
					clip_behavior=None,
			),
			alignment=ft.alignment.center, #type: ignore
			bgcolor=ft.colors.WHITE10,
			expand=True
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
		row_items.append(ft.Text(value = webui_utils.get_textual_inversion_row_value(row_name)))
		return row_items

	def get_textual_inversion_results_grid():
		grid_rows = []
		for item in webui_utils.textual_inversion_grid_row_list:
			grid_rows.append(
				ft.Row(
					controls = get_textual_inversion_grid_row(item),
					height = 50,
				)
			)
		return ft.Column(controls = grid_rows)

	def get_textual_inversion_results(e):
		e.control.data = get_textual_inversion_settings()
		webui_utils.run_textual_inversion(e.control.data)
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
							ft.Draggable(content=ft.Divider(height=10, color="gray")),
					]
			),
			expand = True
	)

	## property panel
	property_panel = ft.Column(
			width = 400,
			controls = [
				latent_space_layout_properties,
			]
	)


###### main panel #################################################################
	main_panel = ft.Row(
			controls = [
				toolbar,
				ft.VerticalDivider(width=10, color="gray"),
				canvas,
				ft.VerticalDivider(width=10, color="gray"),
				property_panel,
			],
			expand=True,
	)


###### bottom_panel ###############################################################
	status_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)
	message_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)
	timeline_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)
	python_console_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)

	bottom_panel = ft.Row(
			height = 150,
			controls = [
				ft.Tabs(
						selected_index = 0,
						animation_duration = 300,
						tabs = [
							ft.Tab(
									text = "Status",
									content = status_window,
							),
							ft.Tab(
									text = "Messages",
									content = message_window,
							),
							ft.Tab(
									text = "Timeline",
									content = timeline_window,
							),
							ft.Tab(
									text = "Python Console",
									content = python_console_window,
							),
						],
				),
			],
	)


###### make page #########################################################
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
	page.add(main_panel)
	page.add(ft.Draggable(content=ft.Divider(height=10, color="gray")))
	page.add(bottom_panel)
	#page.add(ft.Container(ft.Text("test", selectable=True),height=500))



ft.app(target=main, port=8505, view=ft.WEB_BROWSER)

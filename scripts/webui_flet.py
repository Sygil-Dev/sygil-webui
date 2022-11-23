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
	# main function defines

	def load_settings():
		settings = webui_flet_utils.get_user_settings_from_config()
		page.session.set('settings',settings)
		page.update()

	def save_settings_to_config():
		settings = page.session.get('settings')
		webui_flet_utils.save_user_settings_to_config(settings)

	def reset_settings_from_config():
		settings = webui_flet_utils.get_default_settings_from_config()
		page.session.remove('settings')
		page.session.set('settings',settings)
		save_settings_to_config()


	def change_theme(e):
		page.theme_mode = "dark" if page.theme_mode == "light" else "light"

		if "(Light theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Light theme)", '')

		if "(Dark theme)" in theme_switcher.tooltip:
			theme_switcher.tooltip = theme_switcher.tooltip.replace("(Dark theme)", '')

		theme_switcher.tooltip += "(Light theme)" if page.theme_mode == "light" else "(Dark theme)"
		page.update()


#	init ###############################################################
	if not page.session.contains_key('settings'):
		load_settings()

#	layouts ############################################################
	def change_layout(e):
		current_layout.value = e.control.data
		set_current_layout_options(e.control.data)
		set_current_layout_tools(e.control.data)
		set_property_panel_options(e.control.data)
		page.update()

	def set_current_layout_options(layout):
		for control in current_layout_options.controls:
			 current_layout_options.controls.pop()
		if layout == 'Default':
			current_layout_options.controls.append(default_layout_options)
		elif layout == 'Textual Inversion':
			current_layout_options.controls.append(textual_inversion_layout_options)
		elif layout == 'Node Editor':
			current_layout_options.controls.append(node_editor_layout_options)

	def set_current_layout_tools(layout):
		for control in current_layout_tools.controls:
			 current_layout_tools.controls.pop()
		if layout == 'Default':
			current_layout_tools.controls.append(default_layout_tools)
		elif layout == 'Textual Inversion':
			current_layout_tools.controls.append(textual_inversion_layout_tools)
		elif layout == 'Node Editor':
			current_layout_tools.controls.append(node_editor_layout_tools)

	def set_property_panel_options(layout):
		controls = property_panel.content.controls
		for control in controls:
			 controls.pop()
		if layout == 'Default':
			controls.append(default_layout_properties)
		elif layout == 'Textual Inversion':
			controls.append(textual_inversion_layout_properties)
		elif layout == 'Node Editor':
			controls.append(node_editor_layout_properties)


#	settings window ####################################################
	def close_settings_window(e):
		settings_window.open = False
		page.update()

	def open_settings_window(e):
		page.dialog = settings_window
		settings_window.open = True
		page.update()

	def update_general_settings():
		general_settings = get_general_settings()
		general_settings_page.controls = general_settings

	def update_ui_settings():
		ui_settings = get_ui_settings()
		ui_settings_page.controls = ui_settings

	def update_settings():
		update_general_settings()
		update_ui_settings()
		page.update()

	def save_settings(e):
		save_settings_to_config()
		update_settings()

	def reset_settings(e):
		reset_settings_from_config()
		update_settings()

	def general_setting_changed(e):
		settings = page.session.get('settings')
		settings['general'][e.control.label]['value'] = e.control.value
		update_general_settings()
		page.update()

	def ui_setting_changed(e):
		settings = page.session.get('settings')
		settings['webui'][e.control.label] = e.control.value
		update_ui_settings()
		page.update()

	def get_general_settings():
		settings = page.session.get('settings')
		general_settings = [ft.Divider(height=10, color='gray')]
		for setting in settings['general']:
			if 'value' not in settings['general'][setting]:
				continue
			display = None
			display_type = settings['general'][setting]['display']
			if display_type == 'dropdown':
				option_list = []
				for i in range(len(settings['general'][setting]['option_list'])):
					item = ft.dropdown.Option(
							text = settings['general'][setting]['option_list'][i]
					)
					option_list.append(item)
				display = ft.Dropdown(
						label = setting,
						value = settings['general'][setting]['value'],
						options = option_list,
						on_change = general_setting_changed,
				)
			elif display_type == 'slider':
				display = ft.Slider(
						label = setting,
						value = settings['general'][setting]['value'],
						min = settings['general'][setting]['min'],
						max = settings['general'][setting]['max'],
						on_change = general_setting_changed,
				)
			elif display_type == 'textinput':
				display = ft.TextField(
						label = setting,
						value = settings['general'][setting]['value'],
						on_change = general_setting_changed,
				)
			else:
				continue
			new_row = ft.Row(
					controls = [
							display,
					]
			)
			general_settings.append(new_row)
		return general_settings
		

	def get_ui_settings():
		settings = page.session.get('settings')
		ui_settings = [ft.Divider(height=10, color="gray")]
		for setting in settings['webui']:
			value = ft.TextField(
					label = setting,
					value = settings['webui'][setting],
					on_change = ui_setting_changed,
			)
			new_row = ft.Row(
					controls = [
							value,
					]
			)
			ui_settings.append(new_row)
		return ui_settings

	general_settings = get_general_settings()
	general_settings_page = ft.Column(
			alignment = 'start',
			scroll = 'auto',
			controls = general_settings,
	)

	performance_settings_page = ft.Column(
			alignment = 'start',
			scroll = 'auto',
			controls = [
					ft.Divider(height=10, color="gray"),
			],
	)

	server_settings_page = ft.Column(
			alignment = 'start',
			scroll = True,
			controls = [
					ft.Divider(height=10, color="gray"),
			],
	)

	
	ui_settings = get_ui_settings()
	ui_settings_page = ft.Column(
			alignment = 'start',
			scroll = True,
			controls = ui_settings,
	)

	settings_window = ft.AlertDialog(
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
										content = general_settings_page,
								),
								ft.Tab(
										text = "Performance",
										content = performance_settings_page,
								),
								ft.Tab(
										text = "Server",
										content = server_settings_page,
								),
								ft.Tab(
										text = "Interface",
										content = ui_settings_page,
								),
							],
					),
			),
			actions = [
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
			actions_alignment="end",
			#on_dismiss=lambda e: print("Modal dialog dismissed!"),
	)

#	gallery window #####################################################
	def close_gallery_window(e):
		gallery_window.open = False
		page.update()

	def open_gallery_window(e):
		page.dialog = gallery_window
		gallery_window.open = True
		page.update()

	gallery_window = ft.AlertDialog(
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
	        content_padding=10,
	        #text_align="center",
			shift_enter = True,
			tooltip = "Prompt to use for generation.",
			autofocus = True,
			hint_text = "A corgi wearing a top hat as an oil painting.",
			height = 50,
			text_size = 20,
	)

	generate_button = ft.ElevatedButton(
			text = "Generate",
			on_click=None,
			height = 50,
	)

	layouts = ft.Dropdown(
			options = [
				ft.dropdown.Option(text="Default"),
				ft.dropdown.Option(text="Textual Inversion"),
				ft.dropdown.Option(text="Node Editor"),
			],
	        value="Default",
	        content_padding=10,
	        width=200,
	        on_change=change_layout,
			tooltip = "Switch between different workspaces",
			height = 50,
	)

	#current_layout = ft.Text(
			#value = 'Default',
			#size = 20,
			#tooltip = "Current Workspace",
	#)

	layout_menu = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(content = layouts),
				#ft.Container(content = current_layout),
			],
			height = 50,
	)

	default_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Text(value = 'Canvas'), tooltip ='Canvas Options', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Layers'), tooltip ='Layer Options', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Tools'), tooltip ='Toolbox', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Text(value = 'Preferences'), tooltip ='Set Editor Preferences', on_click = None, disabled=True)),
			],
			height = 50,
	)

	textual_inversion_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='textual_inversion options 1', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 2', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'textual_inversion options 3', on_click = None, disabled=True)),
			],
			height = 50,
	)

	node_editor_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='node_editor options 1', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 2', on_click = None, disabled=True)),
				ft.Container(ft.IconButton(content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'node_editor options 3', on_click = None, disabled=True)),
			],
			height = 50,
	)

	current_layout_options = ft.Row(
			alignment = 'start',
			controls = [
				ft.Container(content = default_layout_options),
			],
			height = 50,
	)

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

	#menu_button = ft.PopupMenuButton(
			#items = [
					##ft.PopupMenuItem(text="Settings", on_click=open_settings_modal),
					#ft.PopupMenuItem(),  # divider
					##ft.PopupMenuItem(text="Checked item", checked=False, on_click=check_item_clicked),
			#],
			#height = 50,
	#)

	option_bar = ft.Row(
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
					ft.Container(expand = 4, content = option_bar),
			],
			height = 50,
	)


	# toolbar ############################################################
	open_gallery_button = ft.IconButton(width = 50, content = ft.Icon(ft.icons.DASHBOARD_OUTLINED), tooltip = 'Gallery', on_click = open_gallery_window)
	import_image_button = ft.IconButton(width = 50, content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip = 'Import image as new layer', on_click = None)

	universal_tools = ft.Row(
			alignment = 'start',
			width = 50,
			wrap = True,
			controls = [
				open_gallery_button,
				import_image_button,
			]
	)

	# default layout tools
	default_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	# textual inversion tools
	textual_inversion_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	# node editor tools
	node_editor_layout_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	current_layout_tools = ft.Column(
			controls = [
				default_layout_tools,
			],
	)

	toolbar = ft.Column(
			controls = [
				ft.Container(content = universal_tools),
				ft.Container(content = current_layout_tools),
			],
	)


	# layers panel #######################################################
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
			layer_icon.data.update({'parent':layer_button})  # <--see what i did there? :)
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
	                        ft.Text("Under Construction.")
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
			content = ft.Text("Under Construction."),
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
	video_editor_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)
	messages_window = ft.Container(bgcolor=ft.colors.BLACK12, height=250)

	bottom_panel = ft.Row(
			height = 150,
			controls = [
				ft.Tabs(
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
				),
			],
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
	        content_padding=10,
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
	        content_padding=10,
			value = "k_lms",
			tooltip = "Sampling method or scheduler to use, different sampling method"
					" or schedulers behave differently giving better or worst performance in more or less steps."
					"Try to find the best one for your needs and hardware.",
	)

	default_layout_properties = ft.Container(
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
									ft.TextField(label="Width", value=512, height=50, expand=1, content_padding=10, suffix_text="W", text_align='center', tooltip="Widgth in pixels.", keyboard_type="number"),
									ft.TextField(label="Height", value=512, height=50, expand=1, content_padding=10, suffix_text="H", text_align='center', tooltip="Height in pixels.",keyboard_type="number"),
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Row(
								controls = [
									ft.TextField(label="CFG", value=7.5, height=50, expand=1, content_padding=10, text_align='center', #suffix_text="CFG",
										tooltip="Classifier Free Guidance Scale.", keyboard_type="number"),
									ft.TextField(label="Sampling Steps", value=30, height=50, expand=1, content_padding=10, text_align='center', tooltip="Sampling steps.", keyboard_type="number"),
								],
								spacing = 4,
								alignment = 'spaceAround',
							),
							ft.Row(
								controls = [
									ft.TextField(
											label = "Seed",
											hint_text = "blank=random seed",
											height = 50,
											expand = 1,
	                                        content_padding=10,
											text_align = 'start',
											#suffix_text = "seed",
											tooltip = "Seed used for the generation, leave empty or use -1 for a random seed. You can also use word as seeds.",
											keyboard_type = "number"
									),
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


	# textual inversion layout properties
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


	# node editor layout properties
	node_editor_layout_properties = ft.Container(
			content = ft.Column(
					controls = [
					]
			),
			expand = True
	)

	# property panel
	property_panel = ft.Container(
			bgcolor = ft.colors.WHITE10,
			content = ft.Column(
					controls = [
							ft.Divider(height=10, opacity = 0),
							default_layout_properties,
					],
			),
	)
	
	# advanced panel
	advanced_panel = ft.Container(
			bgcolor = ft.colors.WHITE10,
			content = ft.Column(
					controls = [
							ft.Divider(height=10, opacity = 0),
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

#	workspace ##########################################################
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


#	make page ##########################################################
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
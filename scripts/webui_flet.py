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

	def message(text, err = 0):
		if err:
			text = "ERROR:  " + text
		add_message_to_messages(err,text)
		webui_flet_utils.log_message(text)

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

#	init ###############################################################
	if not page.session.contains_key('settings'):
		load_settings()
		settings = page.session.get('settings')
		if 'webui_page' in settings:
			if 'default_theme' in settings['webui_page']:
				page.theme_mode = settings['webui_page']['default_theme']['value']
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

	def update_settings_window():
		settings_window.content.content.tabs = get_settings_window_tabs()
		page.update()

	def update_settings_window_tab(section):
		for i, tab in enumerate(settings_window.content.content.tabs):
			if section.startswith(tab.text):
				settings_window.content.content.tabs[i].content = get_settings_window_tab_page(section)
				return

	def apply_settings(e):
		update_settings_window()

	def save_settings(e):
		save_settings_to_config()
		update_settings_window()

	def reset_settings(e):
		reset_settings_from_config()
		update_settings_window()

	def settings_window_tab_setting_changed(e):
		settings = page.session.get('settings')
		settings[e.control.data][e.control.label]['value'] = e.control.value
		update_settings_window_tab(e.control.data)
		page.update()

	def settings_window_tab_slider_changed(e):
		settings = page.session.get('settings')
		parent = e.control.data
		settings[parent.data[0]][parent.data[1]]['value'] = e.control.value
		parent.controls[0].value = e.control.value
		parent.controls[1].value = e.control.value
		update_settings_window_tab(parent.data[0])
		page.update()

	def get_settings_window_tab_page_setting_slider(settings,section,setting,display_width):
		setting_slider = []
		label = ft.Text(
				value = setting,
				text_align = 'center',
		)
		row = ft.Row(
			width = display_width,
			data = [section, setting],
			controls = [],
		)
		slider = ft.Slider(
				value = settings[setting]['value'],
				label = "{value}",
				min = settings[setting]['min'],
				max = settings[setting]['max'],
				divisions = int((settings[setting]['max'] - settings[setting]['min']) / settings[setting]['step']),
				on_change = settings_window_tab_slider_changed,
				data = row,
				expand = 4,
		)
		value = ft.TextField(
				value = settings[setting]['value'],
				on_submit = settings_window_tab_slider_changed,
				data = row,
				content_padding = 10,
				expand = 1,
		)
		row.controls.extend([slider,value])
		setting_slider.extend([label,row])
		return setting_slider

	def get_settings_window_tab_settings(section):
		settings = page.session.get('settings')
		settings = settings[section]
		section_settings = [ft.Divider(height=10, color='gray')]
		display_width = (page.width * 0.25) - 5
		for setting in settings:
			if 'value' not in settings[setting]:
				continue
			display = None
			display_type = settings[setting]['display']
			if display_type == 'dropdown':
				option_list = []
				for i in range(len(settings[setting]['option_list'])):
					item = ft.dropdown.Option(
							text = settings[setting]['option_list'][i]
					)
					option_list.append(item)
				display = ft.Dropdown(
						label = setting,
						value = settings[setting]['value'],
						options = option_list,
						on_change = settings_window_tab_setting_changed,
						data = section,
						content_padding = 10,
						width = display_width,
				)
			elif display_type == 'textinput':
				display = ft.TextField(
						label = setting,
						value = settings[setting]['value'],
						on_submit = settings_window_tab_setting_changed,
						data = section,
						content_padding = 10,
						width = display_width,
				)
			elif display_type == 'bool':
				display = ft.Switch(
						label = setting,
						value = settings[setting]['value'],
						on_change = settings_window_tab_setting_changed,
						data = section,
						width = display_width,
				)
			elif display_type == 'slider':
				display = ft.Column(
						controls = get_settings_window_tab_page_setting_slider(settings,section,setting,display_width),
				)
			else:
				continue
			new_row = ft.Row(
					controls = [
							display,
					]
			)
			section_settings.append(new_row)
		return section_settings

	def get_settings_window_tab_page(section):
		settings_window_tab_page = ft.Column(
				alignment = 'start',
				scroll = 'auto',
				controls = get_settings_window_tab_settings(section),
		)
		return settings_window_tab_page

	def get_settings_window_tabs():
		settings = page.session.get('settings')
		tabs = []
		for section in settings:
			if section.endswith('_page'):
				tab = ft.Tab(
					text = section.split('_')[0],
					content = get_settings_window_tab_page(section),
				)
				tabs.append(tab)
		return tabs

	settings_window_tabs = get_settings_window_tabs()

	settings_window = ft.AlertDialog(
			#modal = True,
			title = ft.Text("Settings"),
			content = ft.Container(
					width = page.width * 0.50,
					content = ft.Tabs(
							selected_index = 0,
							animation_duration = 300,
							tabs = settings_window_tabs,
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
							ft.Text('Under Construction.'),
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

#	menu_button = ft.PopupMenuButton(
			#items = [
			#		#ft.PopupMenuItem(text="Settings", on_click=open_settings_modal),
			#		ft.PopupMenuItem(),  # divider
			#		#ft.PopupMenuItem(text="Checked item", checked=False, on_click=check_item_clicked),
			#],
			#height = 50,
#	)

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
		layer_list = layer_manager.data['layer_list']
		layer_slot = make_layer_slot()
		layer_slot.data['image'] = webui_flet_utils.create_blank_image()
		layer_list.append(layer_slot)
		message("added blank layer to canvas")
		update_layer_manager()

	def add_images_as_layers(images):
		layer_list = layer_manager.data['layer_list']
		for img in images:
			layer_slot = make_layer_slot()
			set_layer_slot_name(layer_slot, img.name)
			layer_slot.data['image'] = img.data
			layer_list.append(layer_slot)
			message(f'added "{img.name}" as layer')
		update_layer_manager()

	def pick_images(e: ft.FilePickerResultEvent):
		images = {}
		for f in e.files:
			images.update({f.name:vars(f)})
		images_loaded, images_not_loaded = webui_flet_utils.load_images(images)
		add_images_as_layers(images_loaded)
		if images_not_loaded:
			for img in images_not_loaded:
				message(f'image not loaded: {img}',1)

	def load_images(e):
		add_images_dialog.pick_files(file_type = 'image', allow_multiple = True)

	add_images_dialog = ft.FilePicker(
			on_result = pick_images,
	)

	page.overlay.append(add_images_dialog)

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
			on_click = load_images,
	)

	universal_tools = ft.Row(
			alignment = 'start',
			width = 50,
			wrap = True,
			controls = [
				open_gallery_button,
				add_blank_layer_button,
				add_image_as_layer_button,
			]
	)

	# default tools
	default_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	# textual inversion tools
	textual_inversion_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	# node editor tools
	node_editor_tools = ft.Row(
			alignment = 'start',
			wrap = True,
			controls = [
			],
	)

	def set_tools(control):
		toolbar.content.controls[1] = control
		toolbar.update()

	current_tools = default_tools

	toolbar = ft.Container(
			content = ft.Column(
					controls = [
						universal_tools,
						current_tools,
					],
			),
	)

#	layer manager ######################################################
	def update_layer_manager():
		update_layer_indexes()
		update_active_layer_list()
		layer_manager.update()
				
	def update_active_layer_list():
		layer_manager.data['active_layer_list'] = []
		layer_list = layer_manager.data['layer_list']
		for layer in layer_list:
			if layer.data['type'] == 'slot':
				if layer.content.content.controls[1].data['visible']:
					layer_manager.data['active_layer_list'].append(layer)

	def update_layer_indexes():
		layer_list = layer_manager.data['layer_list']
		index = 0
		for layer in layer_list:
			if layer.data['type'] == 'slot':
				layer.data['index'] = index
				index += 1

	def move_layer_slot(index):
		layer_list = layer_manager.data['layer_list']
		layer_manager.data['layer_being_moved'] = layer_list.pop(index)
		layer_manager.data['layer_last_index'] = index
		update_layer_manager()

	def insert_layer_slot(index):
		layer_list = layer_manager.data['layer_list']
		layer_list.insert(index,layer_manager.data['layer_being_moved'])
		layer_manager.data['layer_being_moved'] = None
		layer_manager.data['layer_last_index'] = -1
		update_layer_manager()

	# layer slot controls
	def set_layer_slot_name(slot, name):
		slot.content.content.controls[1].content.controls[1].value = name

	def show_hide_layer(e):
		parent = e.control.data['parent']
		if parent.data['visible']:
			parent.data['visible'] = False
			parent.opacity = 0.5
			e.control.icon = ft.icons.VISIBILITY_OFF
		else:
			parent.data['visible'] = True
			parent.opacity = 1.0
			e.control.icon = ft.icons.VISIBILITY
		update_active_layer_list()
		parent.update()

	def show_layer_spacer(e):
		if not e.control.data['has_spacer']:
			e.control.data['has_spacer'] = True
			e.control.content.content.controls[0].visible = True
			e.control.update()

	def hide_layer_spacer(e):
		if e.control.data['has_spacer']:
			e.control.data['has_spacer'] = False
			e.control.content.content.controls[0].visible = False
			e.control.update()

	def layer_right_click(e):
		pass

	def layer_slot_will_accept(e):
		if not layer_manager.data['layer_being_moved']:
			return
		layer_list = layer_manager.data['layer_list']
		index = e.control.data['index']
		show_layer_spacer(e)
		update_layer_manager()

	def layer_slot_accept(e):
		if not layer_manager.data['layer_being_moved']:
			return
		layer_list = layer_manager.data['layer_list']
		index = e.control.data['index']
		hide_layer_spacer(e)
		insert_layer_slot(index)

	def layer_slot_leave(e):
		layer_list = layer_manager.data['layer_list']
		index = e.control.data['index']
		hide_layer_spacer(e)
		if layer_manager.data['layer_being_moved']:
			return
		move_layer_slot(index)


	## tab layer controls
	def layer_will_accept(e):
		if not layer_manager.data['layer_being_moved']:
			return
		layer_list = layer_manager.data['layer_list']
		if layer_list:
			if layer_list[-1].data['type'] != 'spacer':
				layer_list.append(make_layer_spacer())
		else:
			layer_list.append(make_layer_spacer())
		update_layer_manager()

	def layer_accept(e):
		if not layer_manager.data['layer_being_moved']:
			return
		layer_list = layer_manager.data['layer_list']
		if layer_list:
			if layer_list[-1].data['type'] == 'spacer':
				layer_list.pop(-1)
		layer_list.append(layer_manager.data['layer_being_moved'])
		layer_manager.data['layer_being_moved'] = None
		update_layer_manager()

	def layer_leave(e):
		if not layer_manager.data['layer_being_moved']:
			return
		layer_list = layer_manager.data['layer_list']
		if layer_list:
			if layer_list[-1].data['type'] == 'spacer':
				layer_list.pop(-1)
		update_layer_manager()

	def make_layer_spacer():
		layer_spacer = ft.Container(
				content = ft.Divider(
						height = 10,
						color = ft.colors.BLACK
				),
				data = {
						'type':'spacer',
				},
		)
		return layer_spacer


	# layer displays
	def make_layer_display():
		try:
			make_layer_display.count += 1
		except Exception:
			make_layer_display.count = 1

		layer_display = ft.Column(
				controls = [
						ft.Container(
								content = ft.Divider(
										height = 10,
										color = ft.colors.BLACK,
								),
								visible = False,
						),
						ft.Container(
								content = ft.Row(
										controls = [],
										
								),
								data = {
										'visible':True,
								},
								bgcolor = ft.colors.WHITE30,
								padding = 4,
						),
				],
				spacing = 0,
		)
		layer_icon = ft.IconButton(
				icon = ft.icons.VISIBILITY,
				tooltip = 'show/hide',
				on_click = show_hide_layer,
				data = {'parent':layer_display.controls[1]},
		)
		layer_label = ft.TextField(
				value = ("layer_" + str(make_layer_display.count)),
				data = {'parent':layer_display.controls[1]},
				content_padding = 10,
				expand = True,
		)
		layer_handle = ft.GestureDetector(
				content = ft.Draggable(
						group = 'layer',
						content = ft.Icon(
								name = ft.icons.DRAG_HANDLE,
								data = {'parent':layer_display.controls[1]},
								tooltip = 'drag to move',
						),
				),
				on_secondary_tap = layer_right_click,
		)
		layer_display.controls[1].content.controls.extend([layer_icon,layer_label,layer_handle])
		return layer_display

	def make_layer_slot():
		layer_slot = ft.DragTarget(
				group = 'layer',
				content = ft.Container(
						content = make_layer_display(),
				),
				on_will_accept = layer_slot_will_accept,
				on_accept = layer_slot_accept,
				on_leave = layer_slot_leave,
				data = {
						'index': -1,
						'type': 'slot',
						'has_spacer': False,
						'image': None,
				}
		)
		return layer_slot

	layer_manager = ft.Container(
			content = ft.DragTarget(
					group = 'layer',
					content = ft.Column(
							spacing = 0,
							scroll = 'hidden',
							controls = [],
					),
					on_will_accept = layer_will_accept,
					on_accept = layer_accept,
					on_leave = layer_leave,
			),
			padding = ft.padding.only(top = 4),
			bgcolor = ft.colors.WHITE10,
			data = {
				'layer_list': [],
				'active_layer_list': [],
				'layer_being_moved': None,
				'layer_last_index': -1,
			},
	)
	layer_manager.data['layer_list'] = layer_manager.content.content.controls


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
			layer_accept(e)

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
	page.theme_mode = "dark"
	page.add(full_page)



ft.app(target=main, port=8505)

# flet_settings_window.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class SettingsWindow(ft.AlertDialog):
	def setup(self,settings):
		self.get_settings_window_tabs(settings)

	def get_settings_window_tab_page_setting_slider(self,settings,section,setting,display_width):
		setting_slider = []
		setting_value = None
		if settings[setting]['value_type'] == 'int':
			setting_value = int(settings[setting]['value'])
		elif settings[setting]['value_type'] == 'float':
			setting_value = float(settings[setting]['value'])
		else:
			setting_value = settings[setting]['value']
		label = ft.Text(
				value = setting,
				text_align = 'center',
		)
		row = SettingsDisplay(
			width = display_width,
			data = [self, section, setting],
			controls = [],
		)
		slider = ft.Slider(
				value = setting_value,
				label = "{value}",
				min = settings[setting]['min'],
				max = settings[setting]['max'],
				divisions = int((settings[setting]['max'] - settings[setting]['min']) / settings[setting]['step']),
				on_change = row.settings_window_tab_slider_changed,
				data = row,
				expand = 4,
		)
		value = ft.TextField(
				value = setting_value,
				on_submit = row.settings_window_tab_slider_changed,
				data = row,
				content_padding = 10,
				expand = 1,
		)
		row.controls.extend([slider,value])
		setting_slider.extend([label,row])
		return setting_slider

	def get_settings_window_tab_settings(self, settings, section):
		settings = settings[section]
		section_settings = [ft.Divider(height=10, color='gray')]
		display_width = (self.content.width * 0.5) - 5
		for setting in settings:
			if 'value' not in settings[setting]:
				continue
			new_row = SettingsDisplay()
			new_row
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
						on_change = new_row.settings_window_tab_setting_changed,
						data = section,
						content_padding = 10,
						width = display_width,
				)
			elif display_type == 'textinput':
				display = ft.TextField(
						label = setting,
						value = settings[setting]['value'],
						on_submit = new_row.settings_window_tab_setting_changed,
						data = section,
						content_padding = 10,
						width = display_width,
				)
			elif display_type == 'bool':
				display = ft.Switch(
						label = setting,
						value = settings[setting]['value'],
						on_change = new_row.settings_window_tab_setting_changed,
						data = section,
						width = display_width,
				)
			elif display_type == 'slider':
				display = ft.Column(
						controls = self.get_settings_window_tab_page_setting_slider(settings,section,setting,display_width),
				)
			else:
				continue
			new_row.data = [self, section, setting]
			new_row.controls.append(display)
			section_settings.append(new_row)
		return section_settings

	def get_settings_window_tab_page(self, settings, section):
		settings_window_tab_page = ft.Column(
				alignment = 'start',
				scroll = 'auto',
				controls = self.get_settings_window_tab_settings(settings, section),
		)
		return settings_window_tab_page

	def get_settings_window_tabs(self, settings):
		tabs = []
		for section in settings:
			if section.endswith('_page'):
				tab = ft.Tab(
					text = section.split('_')[0],
					content = self.get_settings_window_tab_page(settings, section),
				)
				tabs.append(tab)
		self.content.content.tabs = tabs

	def update_settings_window_tab(self, section):
		settings = self.page.session.get('settings')
		for i, tab in enumerate(self.content.content.tabs):
			if section.startswith(tab.text):
				self.content.content.tabs[i].content = self.get_settings_window_tab_page(settings, section)
				return

	def update_settings_window(self):
		self.get_settings_window_tabs(self.page.session.get('settings'))
		self.page.update()


class SettingsDisplay(ft.Row):
	def settings_window_tab_setting_changed(self, e):
		settings = self.page.session.get('settings')
		settings[e.control.data][e.control.label]['value'] = e.control.value
		update_settings_window_tab(e.control.data)
		self.page.update()

	def settings_window_tab_slider_changed(self, e):
		settings = self.page.session.get('settings')
		parent = e.control.data
		setting = settings[parent.data[1]][parent.data[2]]
		setting_value = None
		if setting['value_type'] == 'int':
			setting_value = int(e.control.value)
		elif setting['value_type'] == 'float':
			setting_value = float(e.control.value)
		else:
			setting_value = e.control.value
		setting['value'] = setting_value
		parent.controls[0].value = setting_value
		parent.controls[1].value = str(setting_value)
		parent.data[0].update_settings_window_tab(parent.data[1])
		self.page.update()

def apply_settings(e):
	settings_window.update_settings_window()

def save_settings(e):
	save_settings_to_config()
	settings_window.update_settings_window()

def reset_settings(e):
	reset_settings_from_config()
	settings_window.update_settings_window()

# SettingsWindow == ft.AlertDialog
settings_window = SettingsWindow(
		title = ft.Text("Settings"),
		content = ft.Container(
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


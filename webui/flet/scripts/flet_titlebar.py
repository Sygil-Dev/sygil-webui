# flet_appbar.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class TitleBar(ft.Container):
	def setup(self):
		self.width = self.page.width
		self.height = self.page.titlebar_height

		self.title.size = self.page.titlebar_height * 0.5
		self.title.color = self.page.tertiary_color

		self.prompt.text_size = max(12, self.page.titlebar_height * 0.25)
		self.prompt.focused_border_color = self.page.tertiary_color

		self.layout_menu.controls[0].text_size = self.page.text_size

		self.theme_switcher.size = self.page.titlebar_height
		self.theme_switcher.icon_size = self.page.titlebar_height * 0.5
		self.theme_switcher.tooltip = f"Click to change between the light and dark themes. Current {'(Light theme)' if self.page.theme_mode == 'light' else '(Dark theme)'}"
		self.theme_switcher.on_click = self.page.change_theme_mode

		self.settings_button.size = self.page.titlebar_height
		self.settings_button.icon_size = self.page.titlebar_height * 0.5
		self.settings_button.on_click = self.page.open_settings

	def on_page_change(self):
		self.width = self.page.width
		self.height = self.page.titlebar_height

		self.title.size = self.page.titlebar_height * 0.5
		self.title.color = self.page.tertiary_color

		self.prompt.text_size = max(12, self.page.titlebar_height * 0.25)
		self.prompt.focused_border_color = self.page.tertiary_color

		self.layout_menu.controls[0].text_size = self.page.text_size

		self.theme_switcher.size = self.page.titlebar_height
		self.theme_switcher.icon_size = self.page.titlebar_height * 0.5
		self.theme_switcher.tooltip = f"Click to change between the light and dark themes. Current {'(Light theme)' if self.page.theme_mode == 'light' else '(Dark theme)'}"

		self.settings_button.size = self.page.titlebar_height
		self.settings_button.icon_size = self.page.titlebar_height * 0.5


title = ft.Text(
		value = "  Sygil  ",
		text_align = 'center',
)

prompt = ft.TextField(
		value = "",
		min_lines = 1,
		max_lines = 1,
		content_padding = ft.padding.only(left = 12, top = 0, right = 0, bottom = 0),
		shift_enter = True,
		autofocus = True,
		expand = True,
		tooltip = "Prompt to use for generation.",
		hint_text = "A corgi wearing a top hat as an oil painting.",
)

generate_button = ft.ElevatedButton(
		text = "Generate",
		on_click = None,
)

def set_layout(e):
	e.page.set_layout(e)

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
						text_size = 20,
#						alignment = ft.alignment.center,
						content_padding = ft.padding.only(left = 12, top = 0, right = 0, bottom = 0),
						tooltip = "Switch between different workspaces",
						on_change = set_layout,
				)
		],
)

layout_menu.text_size = layout_menu.controls[0].text_size

theme_switcher = ft.IconButton(
		ft.icons.WB_SUNNY_OUTLINED,
		)

settings_button = ft.IconButton(
		icon = ft.icons.SETTINGS,
)

option_list = ft.Row(
		controls = [
			ft.Container(content = layout_menu),
			ft.Container(content = theme_switcher),
			ft.Container(content = settings_button),
		],
		alignment = 'end'
)


# TitleBar == ft.Container
titlebar = TitleBar(
		content = ft.Row(
				controls = [
					title,
					prompt,
					#generate_button,
					option_list,
				],
		),
)

titlebar.title = title
titlebar.prompt = prompt
titlebar.generate_button = generate_button
titlebar.layout_menu = layout_menu
titlebar.theme_switcher = theme_switcher
titlebar.settings_button = settings_button


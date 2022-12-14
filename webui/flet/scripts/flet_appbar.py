# flet_appbar.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


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
		tooltip = "Prompt to use for generation.",
		autofocus = True,
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
						alignment = ft.alignment.center,
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

appbar = ft.Row(
		controls = [
				ft.Container(content = title),
				ft.Container(expand = True, content = prompt),
				#ft.Container(expand = 1, content = generate_button),
				ft.Container(content = option_list),
		],
)

appbar.title = title
appbar.prompt = prompt
appbar.generate_button = generate_button
appbar.layout_menu = layout_menu
appbar.theme_switcher = theme_switcher
appbar.settings_button = settings_button


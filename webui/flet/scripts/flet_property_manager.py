# flet_property_manager.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class PropertyManager(ft.Container):
	pass

class PropertyPanel(ft.Container):
	pass

property_panel = PropertyPanel(
		content = ft.Column(
				spacing = 0,
				controls = [
						ft.Text("Under Construction"),
				],
		),
)

output_panel = PropertyPanel(
		content = ft.Column(
				spacing = 0,
				controls = [
						ft.Text("Under Construction."),
				],
		),
)

def resize_property_manager(e: ft.DragUpdateEvent):
	e.page.right_panel_width = max(250, e.page.right_panel_width - e.delta_x)
	property_manager.width = e.page.right_panel_width
	e.page.update()

property_manager_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
		drag_interval = 50,
		on_pan_update = resize_property_manager,
		content = ft.VerticalDivider()
)

property_manager = PropertyManager(
		content = ft.Row(
				controls = [
					property_manager_dragbar,
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
														text = 'Output',
														content = output_panel,
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

property_manager.dragbar = property_manager_dragbar
property_manager.property_panel = property_panel
property_manager.output_panel = output_panel


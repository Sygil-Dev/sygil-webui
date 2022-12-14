# flet_property_manager.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class PropertyManager(ft.Container):
	def set_tab_text_size(self, size):
		for tab in self.tabs:
			tab.tab_content.size = size

	def set_tab_bgcolor(self, color):
		for tab in self.tabs:
			tab.content.content.bgcolor = color

	def set_tab_padding(self, padding):
		for tab in self.tabs:
			tab.content.padding = padding

	def set_tab_margin(self, margin):
		for tab in self.tabs:
			tab.content.margin = margin


class PropertyPanel(ft.Container):
	pass

property_panel = PropertyPanel(
		content = ft.Column(
				spacing = 0,
				controls = [
						ft.Text("Under Construction"),
				],
		),
		clip_behavior = 'antiAlias',
)

output_panel = PropertyPanel(
		content = ft.Column(
				spacing = 0,
				controls = [
						ft.Text("Under Construction."),
				],
		),
		clip_behavior = 'antiAlias',
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
														content = property_panel,
														tab_content = ft.Text(
																value = "Properties",
														),
												),
												ft.Tab(
														content = output_panel,
														tab_content = ft.Text(
																value = "Output",
														),
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
		clip_behavior = 'antiAlias',
)

property_manager.tabs = property_manager.content.controls[1].controls[0].tabs
property_manager.dragbar = property_manager_dragbar
property_manager.property_panel = property_panel
property_manager.output_panel = output_panel


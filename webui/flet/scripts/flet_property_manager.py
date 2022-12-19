# flet_property_manager.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class PropertyManager(ft.Container):
	def setup(self):
		self.width = self.page.right_panel_width
		self.bgcolor = self.page.primary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin
		self.set_tab_text_size(self.page.text_size)
		self.set_tab_bgcolor(self.page.secondary_color)
		self.set_tab_padding(self.page.container_padding)
		self.set_tab_margin(self.page.container_margin)
		self.dragbar.content.width = self.page.vertical_divider_width
		self.dragbar.content.color = self.page.tertiary_color
		self.property_panel.preview.width = self.page.right_panel_width
		self.property_panel.preview_dragbar.content.content.height = self.page.divider_height
		self.property_panel.preview_dragbar.content.content.color = self.page.tertiary_color

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

	def set_preview_size(self, width):
		self.property_panel.preview.width = width

	def set_preview_image(self, image):
		self.property_panel.preview.content.src_base64 = flet_utils.convert_image_to_base64(image)
		self.property_panel.update()

class PropertyPanel(ft.Container):
	def resize_preview(self, e):
		self.page.preview_height = max(200, self.page.preview_height + e.delta_y)
		self.preview.height = self.page.preview_height
		self.update()

preview_pane = ft.Container(
		content = ft.Image(
				src_base64 = None,
				gapless_playback = True,
		),
		image_fit = ft.ImageFit.CONTAIN,
		bgcolor = 'black',
		height = 200,
		padding = 0,
		margin = 0,
)

def resize_preview(e):
	property_panel.resize_preview(e)

preview_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_ROW,
		drag_interval = 50,
		on_pan_update = resize_preview,
		content = ft.Container(
				content = ft.Divider(),
				margin = 0,
				padding = 0,
		),
)

property_panel = PropertyPanel(
		content = ft.Column(
				controls = [
					preview_pane,
					preview_dragbar,
				],
		),
)

property_panel.preview = preview_pane
property_panel.preview_dragbar = preview_dragbar

output_panel = PropertyPanel(
		content = ft.Column(
				controls = [
					ft.Text("Under Construction."),
				],
		),
)

def resize_property_manager(e: ft.DragUpdateEvent):
	e.page.right_panel_width = max(250, e.page.right_panel_width - e.delta_x)
	property_manager.width = e.page.right_panel_width
	property_panel.preview.width = e.page.right_panel_width
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
					ft.VerticalDivider(
							width = 4,
							opacity = 0,
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


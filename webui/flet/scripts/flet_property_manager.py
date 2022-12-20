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
		self.property_panel.canvas_properties_dragbar.content.content.height = self.page.divider_height
		self.property_panel.canvas_properties_dragbar.content.content.color = self.page.tertiary_color
		self.property_panel.layer_properties_dragbar.content.content.height = self.page.divider_height
		self.property_panel.layer_properties_dragbar.content.content.color = self.page.tertiary_color

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
		self.preview.height = max(200, self.preview.height + e.delta_y)
		self.update()

	def resize_canvas_properties(self, e):
		self.canvas_properties.height = max(50, self.canvas_properties.height + e.delta_y)
		self.update()

	def resize_layer_properties(self, e):
		self.layer_properties.height = max(50, self.layer_properties.height + e.delta_y)
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

def get_canvas_properties(e):
	return ft.Column(
			controls = [
				ft.Row(
						controls = [
							ft.TextField(
									label = 'Width',
									value = e.page.canvas_size[0],
									text_align = 'center',
									content_padding = 0,
									expand = 1,
							),
							ft.TextField(
									label = 'Height',
									value = e.page.canvas_size[1],
									text_align = 'center',
									content_padding = 0,
									expand = 1,
							),
						],
				),
			],
	)

def open_close_canvas_properties(e):
	if canvas_property_header.open:
		e.control.icon = ft.icons.ARROW_RIGHT
		e.control.icon_color = None
		canvas_property_header.open = False
		canvas_property_header.controls.pop()
		canvas_property_header.update()
	else:
		e.control.icon = ft.icons.ARROW_DROP_DOWN
		e.control.icon_color = e.page.tertiary_color
		canvas_property_header.open = True
		canvas_property_header.controls.append(get_canvas_properties(e))
		canvas_property_header.update()

canvas_property_header = ft.Column(
		controls = [
			ft.TextButton(
					text = "Canvas Properties",
					icon = ft.icons.ARROW_RIGHT,
					on_click = open_close_canvas_properties,
			),
		],
		height = 50,
)

canvas_property_header.open = False

def resize_canvas_properties(e):
	property_panel.resize_canvas_properties(e)

canvas_property_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_ROW,
		drag_interval = 50,
		on_pan_update = resize_canvas_properties,
		content = ft.Container(
				content = ft.Divider(),
				margin = 0,
				padding = 0,
		),
)

def get_layer_properties(e):
	return ft.Column(
			controls = [
				ft.Row(
						controls = [
							ft.TextField(
									label = 'Width',
									value = e.page.active_layer.image.width,
									text_align = 'center',
									content_padding = 0,
									expand = 1,
							),
							ft.TextField(
									label = 'Height',
									value = e.page.active_layer.image_height,
									text_align = 'center',
									content_padding = 0,
									expand = 1,
							),
						],
				),
			],
	)

def open_close_layer_properties(e):
	if layer_property_header.open:
		e.control.icon = ft.icons.ARROW_RIGHT
		e.control.icon_color = None
		layer_property_header.open = False
		layer_property_header.controls.pop()
		layer_property_header.update()
	else:
		e.control.icon = ft.icons.ARROW_DROP_DOWN
		e.control.icon_color = e.page.tertiary_color
		layer_property_header.open = True
		layer_property_header.controls.append(get_layer_properties(e))
		layer_property_header.update()

layer_property_header = ft.Column(
		controls = [
			ft.TextButton(
					text = "layer Properties",
					icon = ft.icons.ARROW_RIGHT,
					on_click = open_close_layer_properties,
					disabled = True,
			),
		],
		height = 50,
)

layer_property_header.open = False

def resize_layer_properties(e):
	property_panel.resize_layer_properties(e)

layer_property_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_ROW,
		drag_interval = 50,
		on_pan_update = resize_layer_properties,
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
					canvas_property_header,
					canvas_property_dragbar,
					layer_property_header,
					layer_property_dragbar,
				],
		),
)

property_panel.preview = preview_pane
property_panel.preview_dragbar = preview_dragbar
property_panel.canvas_properties = canvas_property_header
property_panel.canvas_properties_dragbar = canvas_property_dragbar
property_panel.layer_properties = layer_property_header
property_panel.layer_properties_dragbar = layer_property_dragbar

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


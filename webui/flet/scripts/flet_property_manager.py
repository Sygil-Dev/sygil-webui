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
		self.property_panel.canvas_properties_divider.content.height = self.page.divider_height
		self.property_panel.canvas_properties_divider.content.color = self.page.tertiary_color
		self.property_panel.layer_properties_divider.content.height = self.page.divider_height
		self.property_panel.layer_properties_divider.content.color = self.page.tertiary_color

		self.page.refresh_canvas_preview()
		self.refresh_canvas_properties()

	def on_page_change(self):
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
		self.property_panel.canvas_properties_divider.content.height = self.page.divider_height
		self.property_panel.canvas_properties_divider.content.color = self.page.tertiary_color
		self.property_panel.layer_properties_divider.content.height = self.page.divider_height
		self.property_panel.layer_properties_divider.content.color = self.page.tertiary_color

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

	def refresh_canvas_properties(self):
		self.property_panel.refresh_canvas_properties()

	def refresh_layer_properties(self):
		self.property_panel.refresh_layer_properties()

	def resize_property_manager(self, e: ft.DragUpdateEvent):
		self.page.right_panel_width = max(250, self.page.right_panel_width - e.delta_x)
		self.width = self.page.right_panel_width
		self.property_panel.preview.width = self.page.right_panel_width
		self.page.update()


class PropertyPanel(ft.Container):
	def resize_preview(self, e):
		self.preview.height = max(200, self.preview.height + e.delta_y)
		self.update()

	def refresh_canvas_properties(self):
		self.canvas_properties.controls[0].controls[1].value = self.page.canvas_size[0]
		self.canvas_properties.controls[0].controls[3].value = self.page.canvas_size[1]
		self.canvas_properties.update()

	def refresh_layer_properties(self):
		active = True if self.page.active_layer else False
		if active:
			self.layer_property_header.disabled = False
			self.layer_property_header.open = True
			self.layer_property_header.icon = ft.icons.ARROW_DROP_DOWN
			self.layer_property_header.icon_color = self.page.tertiary_color
			self.layer_properties.visible = True
			self.layer_properties.controls[0].controls[0].value = self.page.active_layer.label.value
			self.layer_properties.controls[1].controls[0].value = self.page.active_layer.image.width
			self.layer_properties.controls[1].controls[1].value = self.page.active_layer.image.height
		else:
			self.layer_property_header.disabled = True
			self.layer_property_header.open = False
			self.layer_property_header.icon = ft.icons.ARROW_RIGHT
			self.layer_property_header.icon_color = None
			self.layer_properties.visible = False
		self.update()

preview_pane = ft.Container(
		content = ft.Image(
				src_base64 = None,
				gapless_playback = True,
		),
		image_fit = 'contain',
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

def open_close_canvas_properties(e):
	if canvas_property_header.open:
		e.control.icon = ft.icons.ARROW_RIGHT
		e.control.icon_color = None
		canvas_property_header.open = False
		canvas_properties.visible = False
		property_panel.update()
	else:
		e.control.icon = ft.icons.ARROW_DROP_DOWN
		e.control.icon_color = e.page.tertiary_color
		canvas_property_header.open = True
		canvas_properties.visible = True
		property_panel.update()

canvas_property_header = ft.Column(
		controls = [
			ft.TextButton(
					text = "Canvas Properties",
					icon = ft.icons.ARROW_RIGHT,
					on_click = open_close_canvas_properties,
			),
		],
)

canvas_property_header.open = False

canvas_properties = ft.Column(
		visible = False,
		controls = [
			ft.Row(
					controls = [
						ft.Text(
								value = 'Width:',
								text_align = 'center',
								no_wrap = True,
								expand = 2,
						),
						ft.Text(
								value = 0,
								text_align = 'start',
								expand = 1,
						),
						ft.Text(
								value = 'Height:',
								text_align = 'start',
								no_wrap = True,
								expand = 2,
						),
						ft.Text(
								value = 0,
								text_align = 'center',
								expand = 1,
						),
					],
			),
		],
)

canvas_property_divider = ft.Container(
		content = ft.Divider(),
		margin = 0,
		padding = 0,
)

def open_close_layer_properties(e):
	if layer_property_header.open:
		e.control.icon = ft.icons.ARROW_RIGHT
		e.control.icon_color = None
		layer_property_header.open = False
		layer_properties.visible = False
		property_panel.update()
	else:
		e.control.icon = ft.icons.ARROW_DROP_DOWN
		e.control.icon_color = e.page.tertiary_color
		layer_property_header.open = True
		layer_properties.visible = True
		property_panel.update()

layer_property_header = ft.TextButton(
		text = "Layer Properties",
		icon = ft.icons.ARROW_RIGHT,
		on_click = open_close_layer_properties,
		disabled = True,
)

layer_property_header.open = False

def update_layer_name(e):
	e.page.active_layer.label.value = e.control.value
	e.page.asset_manager.update()

layer_properties = ft.Column(
		visible = False,
		controls = [
			ft.Row(
					controls = [
						ft.TextField(
								label = 'Layer Name',
								value = '',
								text_align = 'center',
								content_padding = 0,
								expand = 1,
								on_submit = update_layer_name,
						),
					],
			),
			ft.Row(
					controls = [
						ft.TextField(
								label = 'Width',
								value = 0,
								text_align = 'center',
								content_padding = 0,
								expand = 1,
						),
						ft.TextField(
								label = 'Height',
								value = 0,
								text_align = 'center',
								content_padding = 0,
								expand = 1,
						),
					],
			),
		],
)

layer_property_divider = ft.Container(
		content = ft.Divider(),
		margin = 0,
		padding = 0,
)


property_panel = PropertyPanel(
		content = ft.Column(
				controls = [
					preview_pane,
					preview_dragbar,
					canvas_property_header,
					canvas_properties,
					canvas_property_divider,
					layer_property_header,
					layer_properties,
					layer_property_divider,
				],
		),
)

property_panel.preview = preview_pane
property_panel.preview_dragbar = preview_dragbar
property_panel.canvas_property_header = canvas_property_header
property_panel.canvas_properties_divider = canvas_property_divider
property_panel.canvas_properties = canvas_properties
property_panel.layer_property_header = layer_property_header
property_panel.layer_properties = layer_properties
property_panel.layer_properties_divider = layer_property_divider

output_panel = PropertyPanel(
		content = ft.Column(
				controls = [
					ft.Text("Under Construction."),
				],
		),
)

def resize_property_manager(e):
	property_manager.resize_property_manager(e)

def realign_canvas(e):
	e.page.align_canvas()

property_manager_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
		drag_interval = 50,
		on_pan_update = resize_property_manager,
		on_pan_end = realign_canvas,
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


# flet_tool_manager.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


def open_gallery(e):
	e.page.open_gallery(e)

def blank_layer(e):
	e.page.add_blank_layer()

def load_images(e):
	e.page.load_images()

def tool_select(e):
	e.page.set_current_tool(e)


class Action():
	def __init__(self, label, icon, tooltip, on_click):
		self.label = label
		self.icon = icon
		self.tooltip = tooltip
		self.on_click = on_click
		self.disabled = False

action_list = [
	Action('gallery', ft.icons.DASHBOARD_OUTLINED, 'Gallery', open_gallery),
	Action('blank layer', ft.icons.ADD_OUTLINED, 'Add blank layer', blank_layer),
	Action('load image', ft.icons.IMAGE_OUTLINED, 'Load image as layer', load_images),
]


class Tool():
	def __init__(self, label, icon, tooltip):
		self.label = label
		self.icon = icon
		self.tooltip = tooltip
		self.on_click = tool_select
		self.disabled = True

tool_list = [
	Tool('move', ft.icons.OPEN_WITH_OUTLINED, 'Move layer(s)'),
	Tool('select', ft.icons.HIGHLIGHT_ALT_OUTLINED, 'Select tool'),
	Tool('brush', ft.icons.BRUSH_OUTLINED, 'Brush tool'),
	Tool('fill', ft.icons.FORMAT_COLOR_FILL_OUTLINED, 'Fill tool'),
]


class ToolManager(ft.Container):
	def setup(self):
		self.toolbox.get_tools()
		self.width = self.page.tool_manager_width
		self.bgcolor = self.page.primary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin

		self.toolbox.bgcolor = self.page.secondary_color
		self.toolbox.padding = self.page.container_padding
		self.toolbox.margin = self.page.container_margin

		self.tool_divider.height = self.page.divider_height
		self.tool_divider.color = self.page.tertiary_color

		self.tool_properties.bgcolor = self.page.secondary_color
		self.tool_properties.padding = self.page.container_padding
		self.tool_properties.margin = self.page.container_margin

		self.dragbar.width = self.page.vertical_divider_width
		self.dragbar.color = self.page.tertiary_color

	def on_page_change(self):
		self.width = self.page.tool_manager_width
		self.bgcolor = self.page.primary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin

		self.toolbox.bgcolor = self.page.secondary_color
		self.toolbox.padding = self.page.container_padding
		self.toolbox.margin = self.page.container_margin

		self.tool_divider.height = self.page.divider_height
		self.tool_divider.color = self.page.tertiary_color

		self.tool_properties.bgcolor = self.page.secondary_color
		self.tool_properties.padding = self.page.container_padding
		self.tool_properties.margin = self.page.container_margin

		self.dragbar.width = self.page.vertical_divider_width
		self.dragbar.color = self.page.tertiary_color


	def resize_tool_manager(self, e: ft.DragUpdateEvent):
		self.page.tool_manager_width = max(50, self.page.tool_manager_width + e.delta_x)
		tool_manager.width = self.page.tool_manager_width
		self.page.update()

	def resize_toolbox(self, e: ft.DragUpdateEvent):
		min_height = (self.page.tool_manager_button_size * 2)
		self.page.toolbox_height = max(min_height, self.page.toolbox_height + e.delta_y)
		toolbox.height = self.page.toolbox_height
		self.update()

	def enable_tools(self):
		for tool in self.toolbox.content.controls:
			try:
				if tool.on_click == tool_select:
					tool.disabled = False
			except AttributeError:
				continue  # is divider
		self.update()

	def disable_tools(self):
		for tool in self.toolbox.content.controls:
			try:
				if tool.on_click == tool_select:
					tool.disabled = True
			except AttributeError:
				continue  # is divider
		self.update()

	def clear_tools(self):
		self.toolbox.clear_tools()


class ToolBox(ft.Container):
	def get_tools(self):
		for action in action_list:
			self.content.controls.append(self.make_button(action))
		divider = ft.Container(
				content = ft.Divider(
						height = self.page.divider_height,
						color = self.page.tertiary_color,
				),
				margin = 0,
				padding = ft.padding.only(left = 10, top = 0, right = 0, bottom = 0),
		)
		self.content.controls.append(divider)
		for tool in tool_list:
			self.content.controls.append(self.make_button(tool))
		tool_manager.update()

	def make_button(self,button_info):
		button = ft.IconButton(
			width = self.page.icon_size * 2,
			icon_size = self.page.icon_size,
			content = ft.Icon(button_info.icon),
			selected = False,
			selected_icon_color = self.page.tertiary_color,
			tooltip = button_info.tooltip,
			data = {'label':button_info.label},
			on_click = button_info.on_click,
			disabled = button_info.disabled,
		)
		return button

	def clear_tools(self):
		for control in self.content.controls:
			control.selected = False


class ToolPropertyPanel(ft.Container):
	pass


# ToolBox == ft.Container
toolbox = ToolBox(
		clip_behavior = 'antiAlias',
		content = ft.Row(
				alignment = 'start',
				wrap = True,
				spacing = 0,
				run_spacing = 0,
				controls = [],
		),
		margin = 0,
		padding = ft.padding.only(left = 15, top = 0, right = 0, bottom = 0),
)

def resize_toolbox(e):
	tool_manager.resize_toolbox(e)

tool_divider = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_ROW,
		drag_interval = 50,
		on_pan_update = resize_toolbox,
		content = ft.Container(
				content = ft.Divider(),
				margin = 0,
				padding = ft.padding.only(left = 10, top = 0, right = 0, bottom = 0),
		),
)


# ToolPropertyPanel == ft.Container
tool_properties = ToolPropertyPanel(
		content = ft.Column(
				controls = [],
		)
)

def resize_tool_manager(e):
	tool_manager.resize_tool_manager(e)

def realign_canvas(e):
	e.page.align_canvas()

tool_manager_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
		drag_interval = 50,
		on_pan_update = resize_tool_manager,
		on_pan_end = realign_canvas,
		content = ft.VerticalDivider(),
)


# ToolManager = ft.Container
tool_manager = ToolManager(
		content = ft.Row(
				controls = [
					ft.Column(
							controls = [
								toolbox,
								tool_divider,
								tool_properties,
							],
							alignment = 'start',
							expand = True,
					),
					tool_manager_dragbar,
				],
				expand = True,
		),
		clip_behavior = 'antiAlias',
)

tool_manager.toolbox = toolbox
tool_manager.tool_divider = tool_divider.content.content
tool_manager.tool_properties = tool_properties
tool_manager.dragbar = tool_manager_dragbar.content


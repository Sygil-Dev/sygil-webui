# flet_tool_manager.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils

def open_gallery(e):
	e.control.page.open_gallery(e)

def blank_layer(e):
	e.control.page.asset_manager.add_blank_layer(e)

def load_image(e):
	e.control.page.file_picker.pick_files(file_type = 'image', allow_multiple = True)

def tool_select(e):
	toolbox.clear_tools()
	e.control.page.current_tool = e.control.data['label']
	e.control.selected = True
	e.control.page.update()


class Action():
	def __init__(self, label, icon, tooltip, on_click):
		self.label = label
		self.icon = icon
		self.tooltip = tooltip
		self.on_click = on_click

action_list = [
	Action('gallery', ft.icons.DASHBOARD_OUTLINED, 'Gallery', open_gallery),
	Action('blank layer', ft.icons.ADD_OUTLINED, 'Add blank layer', blank_layer),
	Action('load image', ft.icons.IMAGE_OUTLINED, 'Load image as layer', load_image),
]

class Tool():
	def __init__(self, label, icon, tooltip, on_click):
		self.label = label
		self.icon = icon
		self.tooltip = tooltip
		self.on_click = on_click

tool_list = [
	Tool('move', ft.icons.OPEN_WITH_OUTLINED, 'Move layer(s)', tool_select),
	Tool('select', ft.icons.HIGHLIGHT_ALT_OUTLINED, 'Select tool', tool_select),
	Tool('brush', ft.icons.BRUSH_OUTLINED, 'Brush tool', tool_select),
	Tool('fill', ft.icons.FORMAT_COLOR_FILL_OUTLINED, 'Fill tool', tool_select),
]

class ToolBar(ft.Container):
	def setup(self):
		self.toolbox.get_tools()

	def resize_toolbar(self, e: ft.DragUpdateEvent):
		self.page.toolbar_width = max(50, self.page.toolbar_width + e.delta_x)
		toolbar.width = self.page.toolbar_width
		self.page.update()

	def resize_toolbox(self, e: ft.DragUpdateEvent):
		min_height = (self.page.toolbar_button_size * 2)
		self.page.toolbox_height = max(min_height, self.page.toolbox_height + e.delta_y)
		toolbox.height = self.page.toolbox_height
		self.update()

class ToolBox(ft.Container):
	def get_tools(self):
		for action in action_list:
			self.content.controls.append(self.make_button(action))
		divider = ft.Divider(
				height = self.page.divider_height,
				color = self.page.tertiary_color,
		)
		self.content.controls.append(divider)
		for tool in tool_list:
			self.content.controls.append(self.make_button(tool))
		toolbar.update()

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
		)
)

def resize_toolbox(e):
	toolbar.resize_toolbox(e)

tool_divider = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_ROW,
		drag_interval = 50,
		on_pan_update = resize_toolbox,
		content = ft.Divider(),
)

# ToolPropertyPanel == ft.Container
tool_properties = ToolPropertyPanel(
		content = ft.Column(
				controls = [],
		)
)

def resize_toolbar(e):
	toolbar.resize_toolbar(e)

toolbar_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
		drag_interval = 50,
		on_pan_update = resize_toolbar,
		content = ft.VerticalDivider(),
)

# ToolBar = ft.Container
toolbar = ToolBar(
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
					toolbar_dragbar,
				],
				expand = True,
		),
		clip_behavior = 'antiAlias',
)

toolbar.toolbox = toolbox
toolbar.tool_divider = tool_divider
toolbar.tool_properties = tool_properties
toolbar.dragbar = toolbar_dragbar


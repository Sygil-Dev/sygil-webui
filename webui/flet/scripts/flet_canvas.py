# flet_canvas.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class Canvas(ft.Container):
	def setup(self):
		self.refresh_canvas()

	def refresh_canvas(self):
		self.overlay.refresh_canvas_overlay()

	def set_current_tool(self,tool):
		self.page.current_tool = tool

	def center_canvas(self):
		self.image_stack.left = (self.page.workspace_width * 0.5) - (self.page.canvas_size[0] * 0.5),
		self.image_stack.top = (self.page.workspace_height * 0.5) - (self.page.canvas_size[1] * 0.5),
		canvas.update()

	def pan_canvas(self):
		pass

	def zoom_canvas(self):
		pass


class CanvasOverlay(ft.Stack):
	def refresh_canvas_overlay(self):
		self.refresh_canvas_size_display()

	def refresh_canvas_size_display(self):
		self.canvas_size.content.value = str(self.page.canvas_size)
		self.update()

class ImageStack(ft.GestureDetector):
	def add_layer_image(self):
		pass


class LayerImage(ft.GestureDetector):
	def center_layer(self):
		self.left = (self.page.workspace_width * 0.5) - (self.page.canvas_size[0] * 0.5),
		self.top = (self.page.workspace_height * 0.5) - (self.page.canvas_size[1] * 0.5),
		canvas.update()

	def drag_layer(self):
		pass

	def resize_layer(self):
		pass

	def draw_on_layer(self):
		pass

	def paint_on_layer(self):
		pass


image_stack = ImageStack(
		mouse_cursor = ft.MouseCursor.GRAB,
		drag_interval = 50,
		on_pan_update = None,
		on_scroll = None,
		left = 0,
		top = 0,
		content = ft.Stack(),
)

def zoom_in_canvas(e):
	pass

zoom_in_button = ft.IconButton(
		content = ft.Icon(ft.icons.ZOOM_IN_OUTLINED),
		tooltip = 'zoom in canvas',
		on_click = zoom_in_canvas,
)

def zoom_out_canvas(e):
	pass

zoom_out_button = ft.IconButton(
		content = ft.Icon(ft.icons.ZOOM_OUT_OUTLINED),
		tooltip = 'zoom out canvas',
		on_click = zoom_out_canvas,
)

canvas_tools = ft.Container(
		content = ft.Column(
				controls = [
					zoom_in_button,
					zoom_out_button,
				],
		),
		top = 4,
		right = 4,
)

canvas_tools.zoom_in = zoom_in_button
canvas_tools.zoom_out = zoom_out_button

canvas_size_display = ft.Container(
		content = ft.Text(
				value = "test",
		),
		bottom = 4,
		right = 4,
)

canvas_overlay = CanvasOverlay(
		[
			canvas_tools,
			canvas_size_display,
		],
)

canvas_overlay.tools = canvas_tools
canvas_overlay.canvas_size = canvas_size_display

canvas = Canvas(
		content = ft.Stack(
				[
					image_stack,
					canvas_overlay,
				],
				clip_behavior = None,
		),
		alignment = ft.alignment.center,
		expand = True,
)

canvas.image_stack = image_stack
canvas.overlay = canvas_overlay


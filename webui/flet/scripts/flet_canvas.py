# flet_canvas.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class Canvas(ft.Container):
	def setup(self):
		self.bgcolor = self.page.secondary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin

		self.overlay.tools.zoom_in = self.page.icon_size
		self.overlay.tools.zoom_out = self.page.icon_size

		self.overlay.canvas_size.content.color = self.page.text_color
		self.overlay.canvas_size.content.size = self.page.text_size
		self.add_canvas_background()
		self.center_canvas()
		self.refresh_canvas()

	def refresh_canvas(self):
		self.overlay.refresh_canvas_overlay()

	def set_current_tool(self,tool):
		self.page.current_tool = tool

	def add_canvas_background(self):
		self.image_stack.add_canvas_background()
		self.update()

	def add_layer_image(self, image):
		self.image_stack.add_layer_image(image)
		self.update()

	def center_canvas(self):
		width, height = self.page.get_viewport_size()
		self.image_stack.left = (width * 0.5) - (self.image_stack.width * 0.5)
		self.image_stack.top = (height * 0.5) - (self.image_stack.height * 0.5)
		self.update()

	def pan_canvas(self, e):
		pass

	def zoom_in(self, e):
		if self.image_stack.scale >= 4.0:
			self.image_stack.scale = 4.0
		else:
			self.image_stack.scale += 0.05
		self.center_canvas()

	def zoom_out(self, e):
		if self.image_stack.scale <= 0.1:
			self.image_stack.scale = 0.1
		else:
			self.image_stack.scale -= 0.05
		self.center_canvas()

class CanvasOverlay(ft.Stack):
	def refresh_canvas_overlay(self):
		self.refresh_canvas_size_display()

	def refresh_canvas_size_display(self):
		self.canvas_size.content.value = str(self.page.canvas_size)
		self.update()

def pan_canvas(e):
	canvas.pan_canvas(e)

class ImageStack(ft.GestureDetector):
	def add_canvas_background(self):
		image = self.page.canvas_background
		canvas_bg = LayerImage(
				mouse_cursor = None,
				drag_interval = 50,
				on_pan_update = pan_canvas,
				left = 0,
				top = 0,
				width = self.width,
				height = self.height,
				content = ft.Image(
						src_base64 = flet_utils.convert_image_to_base64(image),
						width = 256,
						height = 256,
						repeat = 'repeat',
						gapless_playback = True,
				),
		)
		canvas_offset_x = 0
		canvas_offset_y = 0
		self.canvas_bg = canvas_bg
		self.content.controls.append(canvas_bg)

	def add_layer_image(self, image):
		layer_image = LayerImage(
				mouse_cursor = ft.MouseCursor.GRAB,
				drag_interval = 50,
				on_pan_update = self.drag_layer,
				left = 0,
				top = 0,
				width = image.width,
				height = image.height,
				content = ft.Image(
						src_base64 = flet_utils.convert_image_to_base64(image),
						width = image.width,
						height = image.height,
						gapless_playback = True,
				),
		)
		layer_image.offset_x = 0
		layer_image.offset_y = 0
		self.content.controls.append(layer_image)

	def center_layer(self, e):
		width, height = self.page.get_viewport_size()
		e.control.left = (width * 0.5) - (e.control.width * 0.5)
		e.control.top = (height * 0.5) - (e.control.height * 0.5)
		e.control.offset_x = 0
		e.control.offset_y = 0
		canvas.update()

	def drag_layer(self, e):
		pass

	def resize_layer(self, e):
		pass

	def draw_on_layer(self, e):
		pass

	def paint_on_layer(self, e):
		pass

class LayerImage(ft.GestureDetector):
	pass

def pan_canvas(e):
	canvas.pan_canvas(e)

image_stack = ImageStack(
#		mouse_cursor = ft.MouseCursor.GRAB,
		drag_interval = 50,
		on_pan_update = pan_canvas,
		width = 4096,
		height = 4096,
		left = 0,
		top = 0,
		scale = 1.0,
		content = ft.Stack(),
)

image_stack.offset_x = 0
image_stack.offset_y = 0

canvas_size_display = ft.Container(
		content = ft.Text(
				value = "test",
		),
)

def zoom_in_canvas(e):
	canvas.zoom_in(e)

zoom_in_button = ft.IconButton(
		content = ft.Icon(ft.icons.ZOOM_IN_OUTLINED),
		tooltip = 'zoom in canvas',
		on_click = zoom_in_canvas,
)

def zoom_out_canvas(e):
	canvas.zoom_out(e)

zoom_out_button = ft.IconButton(
		content = ft.Icon(ft.icons.ZOOM_OUT_OUTLINED),
		tooltip = 'zoom out canvas',
		on_click = zoom_out_canvas,
)

canvas_tools = ft.Container(
		content = ft.Column(
				controls = [
					canvas_size_display,
					zoom_in_button,
					zoom_out_button,
				],
				horizontal_alignment = 'end',
		),
		top = 4,
		right = 4,
		padding = 4,
		border_radius = 10,
		opacity = 0.5,
		bgcolor = 'black',

)

canvas_tools.zoom_in = zoom_in_button
canvas_tools.zoom_out = zoom_out_button

canvas_overlay = CanvasOverlay(
		[
			canvas_tools,
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
		),
		clip_behavior = 'antiAlias',
		alignment = ft.alignment.center,
		expand = True,
)

canvas.image_stack = image_stack
canvas.overlay = canvas_overlay


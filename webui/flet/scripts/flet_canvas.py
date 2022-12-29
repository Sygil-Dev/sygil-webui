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

		self.overlay.tools.center.icon_size = self.page.icon_size
		self.overlay.tools.zoom_in.icon_size = self.page.icon_size
		self.overlay.tools.zoom_out.icon_size = self.page.icon_size

		self.overlay.size_display.content.color = self.page.text_color
		self.overlay.size_display.content.size = self.page.text_size
		self.add_canvas_background()
		self.center_canvas()
		self.refresh_canvas()

	def on_page_change(self):
		self.bgcolor = self.page.secondary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin

		self.overlay.tools.center.icon_size = self.page.icon_size
		self.overlay.tools.zoom_in.icon_size = self.page.icon_size
		self.overlay.tools.zoom_out.icon_size = self.page.icon_size

		self.overlay.size_display.content.color = self.page.text_color
		self.overlay.size_display.content.size = self.page.text_size
		self.refresh_canvas()

	def refresh_canvas(self):
		self.image_stack.refresh_stack()
		self.align_canvas()
		self.overlay.refresh_canvas_overlay()

	def set_current_tool(self, tool):
		self.page.current_tool = tool

	def add_canvas_background(self):
		self.image_stack.add_canvas_background()

	def add_layer_image(self, image):
		return self.image_stack.add_layer_image(image)

	def get_image_stack_preview(self):
		return self.image_stack.get_preview()

	def center_canvas(self):
		width, height = self.page.get_viewport_size()
		self.image_stack.offset_x = 0
		self.image_stack.offset_y = 0
		self.image_stack.left = (width * 0.5) - (self.image_stack.width * 0.5)
		self.image_stack.top = (height * 0.5) - (self.image_stack.height * 0.5)
		self.overlay.frame.left = self.image_stack.left
		self.overlay.frame.top = self.image_stack.top
		self.update()

	def align_canvas(self):
		width, height = self.page.get_viewport_size()
		self.image_stack.left = (width * 0.5) - (self.image_stack.width * 0.5) + self.image_stack.offset_x
		self.image_stack.top = (height * 0.5) - (self.image_stack.height * 0.5) + self.image_stack.offset_y
		self.overlay.frame.left = self.image_stack.left
		self.overlay.frame.top = self.image_stack.top
		self.overlay.frame.scale = self.image_stack.scale
		self.update()

	def pan_canvas(self, e: ft.DragUpdateEvent):
		self.image_stack.offset_x += e.delta_x
		self.image_stack.offset_y += e.delta_y
		width, height = self.page.get_viewport_size()
		self.image_stack.offset_x = max(self.image_stack.offset_x, (width - self.image_stack.width) * 0.5)
		self.image_stack.offset_y = max(self.image_stack.offset_y, (height - self.image_stack.height) * 0.5)
		self.image_stack.offset_x = min(self.image_stack.offset_x, (self.image_stack.width - width) * 0.5)
		self.image_stack.offset_y = min(self.image_stack.offset_y, (self.image_stack.height - height) * 0.5)
		self.align_canvas()

	def zoom_in(self, e):
		if self.image_stack.scale >= 4.0:
			self.image_stack.scale = 4.0
		else:
			self.image_stack.scale += 0.05
		self.image_stack.get_scaled_size()
		self.overlay.frame.scale = self.image_stack.scale
		self.align_canvas()

	def zoom_out(self, e):
		if self.image_stack.scale <= 0.1:
			self.image_stack.scale = 0.1
		else:
			self.image_stack.scale -= 0.05
		self.overlay.frame.scale = self.image_stack.scale
		self.image_stack.get_scaled_size()
		self.align_canvas()

	def clear_tools(self):
		self.overlay.clear_tools()

	def set_current_tool(self, tool):
		if tool == 'pan':
			self.overlay.controls.pop(2)
			self.overlay.controls.insert(2,pan_tool)
		elif tool == 'move':
			self.overlay.controls.pop(2)
			self.overlay.controls.insert(2,move_tool)
		elif tool == 'box_select':
			self.overlay.controls.pop(2)
			self.overlay.controls.insert(2,box_select_tool)
		elif tool == 'brush':
			self.overlay.controls.pop(2)
			self.overlay.controls.insert(2,brush_tool)
		elif tool == 'fill':
			self.overlay.controls.pop(2)
			self.overlay.controls.insert(2,fill_tool)
		else:
			pass
		self.update()


class ImageStack(ft.Container):
	def add_canvas_background(self):
		image = self.page.canvas_background
		canvas_bg = LayerImage(
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
		canvas_bg.image = image
		canvas_bg.offset_x = 0
		canvas_bg.offset_y = 0
		self.canvas_bg = canvas_bg
		self.content.controls.append(canvas_bg)

	def add_layer_image(self, image):
		layer_image = None
		if image.path == None:
			layer_image = LayerImage(
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
		else:
			layer_image = LayerImage(
					left = 0,
					top = 0,
					width = image.width,
					height = image.height,
					content = ft.Image(
							src = f'{image.path}',
							width = image.width,
							height = image.height,
							gapless_playback = True,
					),
			)
		layer_image.image = image
		layer_image.offset_x = 0
		layer_image.offset_y = 0
		self.center_layer(layer_image)
		self.content.controls.append(layer_image)
		return layer_image

	def get_preview(self):
		stack = self.content.controls
		return flet_utils.get_preview_from_stack(self.page.canvas_size, stack)

	def refresh_stack(self):
		self.content.controls.clear()
		for slot in self.page.visible_layers:
			self.content.controls.insert(0, slot.layer_image)
		self.content.controls.insert(0, self.canvas_bg)
		self.update()

	def get_scaled_size(self):
		self.scaled_width = self.width * self.scale
		self.scaled_height = self.height * self.scale

	def center_layer(self, layer_image):
		layer_image.offset_x = 0
		layer_image.offset_y = 0
		layer_image.left = (self.width * 0.5) - (layer_image.width * 0.5)
		layer_image.top = (self.height * 0.5) - (layer_image.height * 0.5)

	def align_layer(self, layer_image):
		layer_image.left = ((self.width - layer_image.width) * 0.5) + layer_image.offset_x
		layer_image.top = ((self.height - layer_image.height) * 0.5) + layer_image.offset_y

	def move_layer(self, e: ft.DragUpdateEvent):
		layer = self.page.active_layer.layer_image
		layer.offset_x += e.delta_x
		layer.offset_y += e.delta_y
		self.align_layer(layer)
		self.update()

	def finish_move_layer(self, e: ft.DragEndEvent):
		canvas.refresh_canvas()

	def resize_layer(self, e: ft.DragUpdateEvent):
		pass

	def box_select(self, e):
		pass

	def bucket_fill(self, e):
		pass


class LayerImage(ft.Container):
	pass


class CanvasGestures(ft.GestureDetector):
	pass


class CanvasOverlay(ft.Stack):
	def refresh_canvas_overlay(self):
		self.refresh_canvas_size_display()
		self.page.refresh_canvas_preview()

	def refresh_canvas_size_display(self):
		self.size_display.content.value = str(self.page.canvas_size)
		self.update()

	def clear_tools(self):
		for tool in canvas_tools.content.controls:
			tool.selected = False


# ImageStack == ft.Container
image_stack = ImageStack(
		width = 4096,
		height = 4096,
		left = 0,
		top = 0,
		scale = 1.0,
		content = ft.Stack(),
)

image_stack.offset_x = 0
image_stack.offset_y = 0
image_stack.scaled_width = image_stack.width
image_stack.scaled_height = image_stack.height

canvas_frame = ft.Container(
		width = 4096,
		height = 4096,
		top = 0,
		left = 0,
		scale = 1.0,
		image_fit = 'cover',
		alignment = ft.alignment.center,
		content = ft.Image(
				src_base64 = flet_utils.get_canvas_frame((512,512)),
				gapless_playback = True,
		),
)

# CanvasGestures == ft.GestureDetector
def pan_canvas(e):
	canvas.pan_canvas(e)

pan_tool = CanvasGestures(
		mouse_cursor = ft.MouseCursor.GRAB,
		drag_interval = 10,
		on_pan_update = pan_canvas,
)

def select_layer(e):
	pass

def move_layer(e):
	image_stack.move_layer(e)

def finish_move_layer(e):
	image_stack.finish_move_layer(e)

move_tool = CanvasGestures(
		mouse_cursor = ft.MouseCursor.MOVE,
		drag_interval = 10,
		on_pan_start = select_layer,
		on_pan_update = move_layer,
		on_pan_end = finish_move_layer,
)

def set_select_start(e):
	pass

def draw_select_box(e):
	pass

def get_box_select(e):
	pass

box_select_tool = CanvasGestures(
		mouse_cursor = ft.MouseCursor.GRAB,
		drag_interval = 10,
		on_pan_start = set_select_start,
		on_pan_update = draw_select_box,
		on_pan_end = get_box_select,
)

def draw_on_layer(e):
	pass

brush_tool = CanvasGestures(
		mouse_cursor = ft.MouseCursor.GRAB,
		drag_interval = 10,
		on_pan_update = draw_on_layer,
)

def fill_selection(e):
	pass

fill_tool = CanvasGestures(
		mouse_cursor = ft.MouseCursor.GRAB,
		drag_interval = 10,
		on_tap = fill_selection,
)

canvas_size_display = ft.Container(
		content = ft.Text(
				value = "test",
		),
		left = 4,
		bottom = 4,
		padding = 4,
		border_radius = 10,
		opacity = 0.5,
		bgcolor = 'black',
)

def center_canvas(e):
	canvas.center_canvas()

center_canvas_button = ft.IconButton(
		content = ft.Icon(ft.icons.FILTER_CENTER_FOCUS_OUTLINED),
		tooltip = 'center canvas',
		on_click = center_canvas,
)

def set_pan_tool(e):
	e.page.set_current_tool(e)

pan_canvas_button = ft.IconButton(
		content = ft.Icon(ft.icons.PAN_TOOL_OUTLINED),
		tooltip = 'pan canvas',
		on_click = set_pan_tool,
		selected = True,
		data = {'label':'pan'},
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
					center_canvas_button,
					pan_canvas_button,
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
		disabled = False,
)

canvas_tools.center = center_canvas_button
canvas_tools.zoom_in = zoom_in_button
canvas_tools.zoom_out = zoom_out_button


# CanvasOverlay == ft.Stack
canvas_overlay = CanvasOverlay(
		[
			canvas_frame,
			pan_tool,
			canvas_size_display,
			canvas_tools,
		],
)

canvas_overlay.frame = canvas_frame
canvas_overlay.size_display = canvas_size_display
canvas_overlay.tools = canvas_tools


# Canvas = ft.Container
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


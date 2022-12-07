# flet_canvas.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils



class Canvas(ft.Container):
	def empty_function(self):
		pass


class ImageStack(ft.GestureDetector):
	def center_image_stack(self):
		self.left = (self.page.workspace_width * 0.5) - (self.page.canvas_size[0] * 0.5) - 4,
		self.top = (self.page.workspace_height * 0.5) - (self.page.canvas_size[1] * 0.5) - 4,
		canvas.update()



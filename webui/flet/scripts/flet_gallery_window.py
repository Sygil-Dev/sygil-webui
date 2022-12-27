# flet_gallery_window.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class GalleryWindow(ft.AlertDialog):
	def setup(self):
		self.refresh_galleries()

	def refresh_galleries(self):
		self.refresh_gallery('uploads')
		self.refresh_gallery('outputs')

	def refresh_gallery(self, gallery_name):
		index = None
		if gallery_name == 'uploads':
			self.uploads_gallery.get_gallery_display(gallery_name)
		elif gallery_name == 'outputs':
			self.outputs_gallery.get_gallery_display(gallery_name)
		else:
			page.message(f'{gallery_name} gallery not found.', 1)
			return None

	def get_gallery_images(self, gallery_name):
		return flet_utils.get_gallery_images(gallery_name)

	def select_image(self, e):
		if e.control.border :
			e.control.border = None
			if e.control.image in self.selected_images:
				self.selected_images.remove(e.control.image)
			e.control.update()
		else:
			e.control.border = ft.border.all(2, e.page.tertiary_color)
			self.selected_images.append(e.control.image)
			e.control.update()

class GalleryDisplay(ft.Container):
	def get_gallery_display(self, gallery_name):
		self.content = ft.GridView(
				controls = None,
				padding = 0,
				runs_count = 3,
				run_spacing = 12,
				spacing = 12,
				expand = True,
		)
		gallery = gallery_window.get_gallery_images(gallery_name)
		if not gallery:
			self.content.controls.append(
					ft.Image(
							src = '/images/chickens.jpg',
							tooltip = 'Nothing here but us chickens!',
							gapless_playback = True,
					)
			)
			return

		for image in gallery:
			gallery_image = GalleryImage(
					content = ft.Image(
							src = image.path,
							tooltip = image.filename,
							width = image.width,
							height = image.height,
							gapless_playback = True,
					),
					image_fit = 'contain',
					height = image.height,
					width = image.width,
					padding = 0,
					margin = 0,
					border = None,
					on_click = gallery_window.select_image
			)
			gallery_image.image = image
			self.content.controls.append(gallery_image)

class GalleryImage(ft.Container):
	pass

def add_as_new_layer(e):
	if gallery_window.selected_images:
		e.page.add_images_as_layers(gallery_window.selected_images)
		gallery_window.selected_images.clear()
		for tab in gallery_window.content.content.tabs:
			for image in tab.content.content.controls:
				image.border = None
				image.update()

def save_to_disk(e):
	pass

def remove_from_gallery(e):
	pass

uploads_gallery = GalleryDisplay(
		content = None,
		clip_behavior = 'antiAlias',
)

outputs_gallery = GalleryDisplay(
		content = None,
		clip_behavior = 'antiAlias',
)

# GalleryWindow == ft.AlertDialog
gallery_window = GalleryWindow(
		title = ft.Text('Gallery'),
		content = ft.Container(
				content = ft.Tabs(
						selected_index = 0,
						animation_duration = 300,
						tabs = [
							ft.Tab(
									text = "Uploads",
									content = uploads_gallery,
							),
							ft.Tab(
									text = "Outputs",
									content = outputs_gallery,
							),
						],
				),
		),
		actions = [
				ft.ElevatedButton(
						text = "Add As New Layer(s)",
						icon = ft.icons.ADD_OUTLINED,
						on_click = add_as_new_layer,
				),
				ft.ElevatedButton(
						text = "Save",
						icon = ft.icons.SAVE_OUTLINED,
						on_click = save_to_disk,
				),
				ft.ElevatedButton(
						text = "Discard",
						icon = ft.icons.DELETE_OUTLINED,
						on_click = remove_from_gallery,
				),
		],
		actions_alignment="end",
)

gallery_window.uploads_gallery = uploads_gallery
gallery_window.outputs_gallery = outputs_gallery
gallery_window.selected_images = []

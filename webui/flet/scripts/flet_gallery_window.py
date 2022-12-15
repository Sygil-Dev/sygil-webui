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


class GalleryDisplay(ft.Container):
	def get_gallery_display(self, gallery_name):
		self.content = ft.Stack(
				[
						ft.Row(
								controls = None,
								wrap = False,
								scroll = 'always',
								expand = True,
						),
						ft.Column(
								controls = [
										ft.Row(
												controls = [
														ft.IconButton(
																height = self.height * 0.5,
																icon_size = 50,
																content = ft.Icon(ft.icons.ARROW_CIRCLE_LEFT_OUTLINED),
																tooltip = 'previous image',
																on_click = None,
														),
														ft.IconButton(
																height = self.height * 0.5,
																icon_size = 50,
																content = ft.Icon(ft.icons.ARROW_CIRCLE_RIGHT_OUTLINED),
																tooltip = 'next image',
																on_click = None,
														),
												],
												expand = True,
												alignment = 'spaceBetween',
										),
								],
								alignment = 'center',
						),
				],
		)
		gallery = gallery_window.get_gallery_images(gallery_name)
		if not gallery:
			self.content.controls[0].controls.append(
					ft.Image(
							src = '/images/chickens.jpg',
							tooltip = 'Nothing here but us chickens!',
							gapless_playback = True,
					)
			)
			return

		for i in range(len(gallery)):
			image = gallery[i]
			image_name = list(image.keys())[0]
			image_path = image[image_name]['img_path']
			image_data = None
			if 'info_path' in image[image_name]:
				image_data = image[image_name]['info_path']
			self.content.controls[0].controls.append(
					ft.Image(
							src = image_path,
							tooltip = image_name,
							gapless_playback = True,
					)
			)


def add_as_new_layer(e):
	pass

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
						text = "Add As New Layer",
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


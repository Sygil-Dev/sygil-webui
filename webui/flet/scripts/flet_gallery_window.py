# flet_gallery_window.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class GalleryWindow(ft.AlertDialog):
	def empty(self):
		pass

	def get_gallery_images(self, gallery_name):
		return flet_utils.get_gallery_images(gallery_name)

	def refresh_gallery(self, gallery_name):
		index = None
		if gallery_name == 'uploads':
			index = 0
		elif gallery_name == 'outputs':
			index = 1
		else:
			page.message(f'{gallery_name} gallery not found.', 1)
			return None
		gallery_window.content.content.tabs[index].content = get_gallery_display(gallery_name)



class GalleryDisplay(ft.Container):
	def get_gallery_display(self, gallery_name):
		gallery_display = ft.Stack(
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
																height = page.height * 0.75,
																icon_size = 50,
																content = ft.Icon(ft.icons.ARROW_CIRCLE_LEFT_OUTLINED),
																tooltip = 'previous image',
																on_click = None,
														),
														ft.IconButton(
																height = page.height * 0.75,
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
		gallery = get_gallery_images(gallery_name)
		if len(gallery) < 1:
			gallery_display.controls[0].controls.append(
					ft.Image(
							src = '/images/chickens.jpg',
							tooltip = 'Nothing here but us chickens!',
							gapless_playback = True,
					)
			)
			return gallery_display
			
		for i in range(len(gallery)):
			image = gallery[i]
			image_name = list(image.keys())[0]
			image_path = image[image_name]['img_path']
			image_data = None
			if 'info_path' in image[image_name]:
				image_data = image[image_name]['info_path']
			gallery_display.controls[0].controls.append(
					ft.Image(
							src = image_path,
							tooltip = image_name,
							gapless_playback = True,
					)
			)
		return gallery_display
	

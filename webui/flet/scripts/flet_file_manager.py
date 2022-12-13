# flet_file_manager.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class UploadWindow(ft.AlertDialog):
	def upload_file(self, e):
		if file_picker.result is not None and file_picker.result.files is not None:
			file_list = []
			for f in file_picker.result.files:
				upload_url = e.page.get_upload_url(f.name, 600)
				img = ft.FilePickerUploadFile(f.name,upload_url)
				file_list.append(img)
			file_picker.upload(file_list)

	def upload_complete(self, e):
		e.page.progress_bars.clear()
		e.page.selected_files.controls.clear()
		e.page.close_uploads(e)
		e.page.message('File upload(s) complete.')
		e.page.asset_manager.add_images_as_layers(file_picker.images)
		file_picker.images.clear()
		e.page.refresh_gallery('uploads')

	def get_image_from_uploads(self, name):
		return flet_utils.get_image_from_uploads(name)

	def get_file_display(self, name, progress):
		display = ft.Column(
				controls = [
						ft.Row([ft.Text(name)]),
						progress,
				],
		)
		return display

def upload_file(e):
	uploads.upload_file(e)

def close_upload_window(e):
	e.page.close_uploads(e)

uploads = UploadWindow(
	title = ft.Text("Confirm file upload(s)"),
	content = None,
	actions_alignment = "center",
	actions = [
		ft.ElevatedButton("UPLOAD", on_click = upload_file),
		ft.TextButton("CANCEL", on_click = close_upload_window),
	],
)

class ImportWindow(ft.AlertDialog):
	pass

def import_file(e):
	e.page.close_imports(e)

def close_import_window(e):
	e.page.close_imports(e)

imports = ImportWindow(
	title = ft.Text("Confirm file import(s)"),
	content = None,
	actions_alignment = "center",
	actions = [
		ft.ElevatedButton("IMPORT", on_click = import_file),
		ft.TextButton("CANCEL", on_click = close_import_window),
	],
)

def pick_images(e: ft.FilePickerResultEvent):
	e.page.progress_bars.clear()
	e.page.selected_files.controls.clear()
	# check to see if files or directory were chosen
	if e.files is not None and e.path is None:
		for f in e.files:
			prog = ft.ProgressBar(
					value = 0,
					color = 'blue',
			)
			e.page.progress_bars[f.name] = prog
			e.page.selected_files.controls.append(uploads.get_file_display(f.name,prog))
			file_picker.pending += 1
		# import if local, upload if remote
		if not e.page.web:
			e.page.open_imports(e)
		else:
			e.page.open_uploads(e)

def on_image_upload(e: ft.FilePickerUploadEvent):
	if e.error:
		e.page.message(f"Upload error occurred! Failed to fetch '{e.file_name}'.",1)
		file_picker.pending -= 1
	else:
		# update progress bars
		e.page.progress_bars[e.file_name].value = e.progress
		e.page.progress_bars[e.file_name].update()
		if e.progress >= 1:
			file_picker.pending -= 1
			file_picker.images.update(uploads.get_image_from_uploads(e.file_name))
	if file_picker.pending <= 0:
		file_picker.pending = 0
		uploads.upload_complete(e)

file_picker = ft.FilePicker(
		on_result = pick_images,
		on_upload = on_image_upload
)

file_picker.pending = 0
file_picker.images = {}


# flet_messages.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class Messages(ft.Container):
	def setup(self):
		self.height = self.page.bottom_panel_height
		self.bgcolor = self.page.primary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin

		self.set_tab_text_size(self.page.text_size)
		self.set_tab_bgcolor(self.page.secondary_color)
		self.set_tab_padding(self.page.container_padding)
		self.set_tab_margin(self.page.container_margin)

		self.dragbar.content.height = self.page.divider_height
		self.dragbar.content.color = self.page.tertiary_color

	def on_page_change(self):
		self.height = self.page.bottom_panel_height
		self.bgcolor = self.page.primary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin

		self.set_tab_text_size(self.page.text_size)
		self.set_tab_bgcolor(self.page.secondary_color)
		self.set_tab_padding(self.page.container_padding)
		self.set_tab_margin(self.page.container_margin)

		self.dragbar.content.height = self.page.divider_height
		self.dragbar.content.color = self.page.tertiary_color


	def set_tab_text_size(self, size):
		for tab in self.tabs:
			tab.tab_content.size = size

	def set_tab_bgcolor(self, color):
		for tab in self.tabs:
			tab.content.bgcolor = color

	def set_tab_padding(self, padding):
		for tab in self.tabs:
			tab.content.padding = padding

	def set_tab_margin(self, margin):
		for tab in self.tabs:
			tab.content.margin = margin

	def resize_messages(self, e: ft.DragUpdateEvent):
		self.page.bottom_panel_height = max(100, self.page.bottom_panel_height - e.delta_y)
		self.height = self.page.bottom_panel_height
		self.page.update()

	def message(self, text, err = 0):
		if err:
			text = "ERROR:  " + text
		self.add_message_to_messages(err,text)
		flet_utils.log_message(text)

	def prune_messages(self):
		if len(message_list.controls) > self.page.max_message_history:
			message_list.controls.pop(0)
		message_list.update()

	def add_message_to_messages(self,err,text):
		if err:
			msg = ft.Text(value = text, color = ft.colors.RED)
		else:
			msg = ft.Text(value = text)
		message_list.controls.append(msg)
		self.prune_messages()


message_list = ft.ListView(
		spacing = 4,
		auto_scroll = True,
		controls = [],
)

messages_panel = ft.Container(
		content = message_list,
)

video_editor_panel = ft.Column(
		expand = True,
		controls = [ft.Text("Under Construction")]
)

def resize_messages(e):
	messages.resize_messages(e)

def realign_canvas(e):
	e.page.align_canvas()

messages_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_ROW,
		drag_interval = 50,
		on_pan_update = resize_messages,
		on_pan_end = realign_canvas,
		content = ft.Divider(),
)

messages = Messages(
		content = ft.Stack(
				controls = [
						messages_dragbar,
						ft.Tabs(
								selected_index = 0,
								animation_duration = 300,
								tabs = [
										ft.Tab(
												content = messages_panel,
												tab_content = ft.Text(
														value = 'Messages',
												),
										),
										ft.Tab(
												content = video_editor_panel,
												tab_content = ft.Text(
														value = 'Video Editor',
												),
										),
								],
						),
				],
		),
		clip_behavior = 'antiAlias',
)

messages.dragbar = messages_dragbar
messages.tabs = messages.content.controls[1].tabs
messages.messages_panel = messages_panel
messages.video_editor_panel = video_editor_panel
messages.message_list = message_list


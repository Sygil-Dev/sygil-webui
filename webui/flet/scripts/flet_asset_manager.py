# flet_layer_manager.py

# Flet imports
import flet as ft

# utils imports
from scripts import flet_utils


class AssetManager(ft.Container):
	def setup(self):
		self.width = self.page.left_panel_width
		self.bgcolor = self.page.primary_color
		self.padding = self.page.container_padding
		self.margin = self.page.container_margin

		self.set_tab_text_size(self.page.text_size)
		self.set_tab_bgcolor(self.page.secondary_color)
		self.set_tab_padding(self.page.container_padding)
		self.set_tab_margin(self.page.container_margin)

		self.dragbar.content.width = self.page.vertical_divider_width
		self.dragbar.content.color = self.page.tertiary_color

	def add_blank_layer(self, e):
		self.layer_panel.add_blank_layer(e)

	def add_image_as_layer(self, image):
		self.layer_panel.add_image_as_layer(image)

	def add_images_as_layers(self, images):
		self.layer_panel.add_images_as_layers(images)

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



class AssetPanel(ft.Container):
	pass

class LayerPanel(ft.Container):
	def update_layers(self):
		self.layers = self.content.content.controls
		self.update_layer_indexes()
		self.update_visible_layers()
		self.update()

	def update_layer_indexes(self):
		count = 0
		for layer in self.layers:
			layer.index = count
			count += 1

	def update_visible_layers(self):
		self.visible_layers = []
		for layer in self.layers:
			if layer.visible:
				self.visible_layers.append(layer)

	def update_layer_properties(self,e):
		self.page.property_manager.refresh_layer_properties()

	def make_layer_active(self, index):
		for i, layer in enumerate(self.layers):
			layer.active = False
			layer.handle.color = None
			if i == index:
				layer.active = True
				layer.handle.color = self.page.tertiary_color
				self.page.active_layer = layer
				self.page.property_manager.refresh_layer_properties()

	def add_layer_slot(self, image):
		label = ft.TextField(
				value = image.filename,
				focused_border_color = self.page.tertiary_color,
				text_size = self.page.text_size,
				content_padding = ft.padding.only(left = 12, top = 0, right = 0, bottom = 0),
				expand = True,
				on_submit = self.update_layer_properties,
		)
		handle = ft.Icon(
				name = ft.icons.DRAG_HANDLE,
				color = None,
				tooltip = 'drag to move',
		)
		layer_slot = LayerSlot(
				content = ft.Row(
						controls = [
							label,
							handle,
						],
						height = self.page.layer_height,
						expand = True,
				),
		)
		layer_slot.label = label
		layer_slot.handle = handle
		layer_slot.index = 0
		layer_slot.visible = True
		layer_slot.active = False
		layer_slot.image = image
		self.content.content.controls.insert(0,layer_slot)
		self.layers = self.content.content.controls
		self.update_layer_indexes()
		self.make_layer_active(0)
		self.page.property_manager.refresh_layer_properties()
		self.update()

	def add_blank_layer(self, e):
		image = flet_utils.create_blank_image(self.page.canvas_size)
		self.add_layer_slot(image)
		self.page.canvas.add_layer_image(image)
		self.page.message("added blank layer to canvas")

	def add_image_as_layer(self, image):
		self.add_layer_slot(image)
		self.page.canvas.add_layer_image(image)
		self.page.message(f'added "{image.filename}" as layer')

	def add_images_as_layers(self, images):
		for image in images:
			self.add_image_as_layer(image)

	def get_layer_index_from_position(self, pos):
		index = int(pos / self.page.layer_height)
		if index > len(self.layers):
			return 0
		return index

class LayerSlot(ft.Container):
	pass


def layer_left_click(e: ft.TapEvent):
	index = layer_panel.get_layer_index_from_position(e.local_y)
	layer_panel.make_layer_active(index)
	layer_panel.update()

def pickup_layer(e: ft.DragStartEvent):
	pass

def on_layer_drag(e: ft.DragUpdateEvent):
	pass

def drop_layer(e: ft.DragEndEvent):
	pass

layer_panel = LayerPanel(
		content = ft.GestureDetector(
				content = ft.Column(
						controls = [],
						expand = True,
						scroll = 'hidden',
				),
				drag_interval = 10,
				on_tap_down = layer_left_click,
				on_vertical_drag_start = pickup_layer,
				on_vertical_drag_update = on_layer_drag,
				on_vertical_drag_end = drop_layer,
		),
)

layer_panel.layers = []
layer_panel.visible_layers = []
layer_panel.active_layer = None
layer_panel.layer_being_moved = None
layer_panel.layer_last_index = 0

asset_panel = AssetPanel(
		content = ft.Column(
				controls = [
					ft.Text("Under Construction"),
				],
		),
)

def resize_asset_manager(e: ft.DragUpdateEvent):
	e.page.left_panel_width = max(250, e.page.left_panel_width + e.delta_x)
	asset_manager.width = e.page.left_panel_width
	e.page.update()

asset_manager_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
		drag_interval = 50,
		on_pan_update = resize_asset_manager,
		content = ft.VerticalDivider(),
)

asset_manager = AssetManager(
		content = ft.Row(
				controls = [
					ft.Column(
							controls = [
								ft.Tabs(
										selected_index = 0,
										animation_duration = 300,
										tabs = [
											ft.Tab(
													content = layer_panel,
													tab_content = ft.Text(
															value = "Layers",
													),
											),
											ft.Tab(
													content = asset_panel,
													tab_content = ft.Text(
															value = "Assets",
													),
											),
										],
								),
							],
							alignment = 'start',
							expand = True
					),
					asset_manager_dragbar,
				],
				expand = True,
		),
		clip_behavior = 'antiAlias',
)

asset_manager.tabs = asset_manager.content.controls[0].controls[0].tabs
asset_manager.dragbar = asset_manager_dragbar
asset_manager.layer_panel = layer_panel
asset_manager.asset_panel = asset_panel

'''
	# keep track of which layers are visible
	def show_hide_layer(self, e):
		parent = e.control.data['parent']
		if parent.data['visible']:
			parent.data['visible'] = False
			parent.opacity = 0.5
			e.control.icon = ft.icons.VISIBILITY_OFF
		else:
			parent.data['visible'] = True
			parent.opacity = 1.0
			e.control.icon = ft.icons.VISIBILITY
		self.update_visible_layer_list()
		parent.update()
		self.page.refresh_canvas()

	def update_visible_layer_list(self):
		self.page.visible_layer_list = []
		layer_list = self.page.layer_list
		for layer in layer_list:
			if layer.data['type'] == 'slot':
				if layer.content.content.controls[1].data['visible']:
					self.page.visible_layer_list.append(layer)

	# keep track of which layers are active
	def lock_unlock_layer(self, e):
		parent = e.control.data['parent']
		if parent.data['locked']:
			parent.data['locked'] = False
			e.control.icon = ft.icons.LOCK_OPEN_OUTLINED
		else:
			parent.data['locked'] = True
			e.control.icon = ft.icons.LOCK_OUTLINED
		self.update_active_layer_list()
		parent.update()

	def update_active_layer_list(self):
		self.page.active_layer_list = []
		layer_list = self.page.layer_list
		for layer in layer_list:
			if layer.data['type'] == 'slot':
				if not layer.content.content.controls[1].data['locked']:
					self.page.active_layer_list.append(layer)


'''

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

	def on_page_change(self):
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

		if self.page.active_layer is not None:
			self.page.active_layer.handle.color = self.page.tertiary_color

	def add_image_as_layer(self, image):
		return self.layer_panel.add_image_as_layer(image)

	def add_images_as_layers(self, images):
		return self.layer_panel.add_images_as_layers(images)

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

	def resize_asset_manager(self, e: ft.DragUpdateEvent):
		self.page.left_panel_width = max(250, self.page.left_panel_width + e.delta_x)
		self.width = self.page.left_panel_width
		self.page.update()

	def refresh_layers(self):
		self.layer_panel.refresh_layers()


class AssetPanel(ft.Container):
	pass


class LayerPanel(ft.Container):
	def refresh_layers(self):
		self.layers = self.content.content.controls
		self.refresh_layer_indexes()
		self.refresh_visible_layers()
		self.update()

	def refresh_layer_indexes(self):
		count = 0
		for layer in self.layers:
			layer.index = count
			count += 1

	def refresh_visible_layers(self):
		self.page.visible_layers = []
		for layer in self.layers:
			if not layer.disabled:
				self.page.visible_layers.append(layer)

	def refresh_layer_name(self, e):
		self.page.refresh_layers()

	def make_layer_active(self, index):
		for i, layer in enumerate(self.layers):
			layer.active = False
			layer.handle.color = None
			if i == index:
				layer.active = True
				layer.handle.color = self.page.tertiary_color
				self.page.set_active_layer(layer)

	def add_layer_slot(self, image):
		label = ft.TextField(
				value = image.filename,
				focused_border_color = self.page.tertiary_color,
				text_size = self.page.text_size,
				content_padding = ft.padding.only(left = 12, top = 0, right = 0, bottom = 0),
				expand = True,
				on_submit = self.refresh_layer_name,
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
						expand = True,
				),
				height = self.page.layer_height,
				padding = 0,
				margin = 0,
		)
		layer_slot.label = label
		layer_slot.handle = handle
		layer_slot.index = 0
		layer_slot.disabled = False
		layer_slot.active = False
		layer_slot.image = image
		layer_slot.layer_image = None
		self.content.content.controls.insert(0,layer_slot)
		self.layers = self.content.content.controls
		self.refresh_layer_indexes()
		self.make_layer_active(0)
		return layer_slot

	def add_image_as_layer(self, image):
		return self.add_layer_slot(image)

	def add_images_as_layers(self, images):
		layer_slots = []
		for image in images:
			layer_slots.append(self.add_image_as_layer(image))
		return layer_slots

	def get_layer_index_from_position(self, pos):
		index = int(pos / self.page.layer_height)
		return index

	def move_layer(self, layer, index):
		if index > len(self.layers):
			layer = self.layers.pop(layer.index)
			self.layers.append(layer)
		if layer.index < index:
			index -= 1
		layer = self.layers.pop(layer.index)
		self.layers.insert(index, layer)
		self.page.refresh_layers()

	def delete_layer(self, layer):
		if not layer:
			return
		self.layers.pop(layer.index)


class LayerSlot(ft.Container):
	pass


class LayerActionMenu(ft.Card):
	def show_menu(self):
		self.visible = True
		self.update()

	def hide_menu(self):
		self.visible = False
		self.update()


def close_menu(e):
	layer_action_menu.hide_menu()

def show_hide_layer(e):
	e.page.active_layer.disabled = False if e.page.active_layer.disabled else True
	e.page.refresh_layers()
	close_menu(e)

def move_layer_to_top(e):
	layer_panel.move_layer(e.page.active_layer, 0)
	close_menu(e)

def move_layer_up(e):
	layer_panel.move_layer(e.page.active_layer, e.page.active_layer.index - 1)
	close_menu(e)

def move_layer_down(e):
	layer_panel.move_layer(e.page.active_layer, e.page.active_layer.index + 2)
	close_menu(e)

def delete_layer(e):
	layer_panel.delete_layer(e.page.active_layer)
	e.page.active_layer = None
	e.page.refresh_layers()
	close_menu(e)


class LayerAction():
	def __init__(self, text, on_click):
		self.text = text
		self.on_click = on_click

layer_action_list = [
	LayerAction('Show/Hide Layer', show_hide_layer),
	LayerAction('Move Layer To Top', move_layer_to_top),
	LayerAction('Move Layer Up', move_layer_up),
	LayerAction('Move Layer Down', move_layer_down),
	LayerAction('Delete Layer', delete_layer),
]

def make_action_buttons(action_list):
	button_list = []
	for action in action_list:
		button_list.append(
			ft.TextButton(
					text = action.text,
					on_click = action.on_click,
			)
		)
	return button_list

# LayerActionMenu == ft.Card
layer_action_menu = LayerActionMenu(
		content = ft.GestureDetector(
				content = ft.Column(
						controls = make_action_buttons(layer_action_list),
						expand = False,
						spacing = 0,
						alignment = 'start',
						tight = True,
				),
				on_exit = close_menu,
		),
		margin = 0,
		visible = False,
)



def layer_left_click(e: ft.TapEvent):
	index = layer_panel.get_layer_index_from_position(e.local_y)
	if index >= len(layer_panel.layers):
		return
	layer_panel.make_layer_active(index)
	layer_panel.update()

def layer_right_click(e: ft.TapEvent):
	index = layer_panel.get_layer_index_from_position(e.local_y)
	if index >= len(layer_panel.layers):
		return
	layer_panel.make_layer_active(index)
	layer_panel.update()
	layer_action_menu.left = e.global_x
	layer_action_menu.top = e.global_y
	layer_action_menu.show_menu()

def pickup_layer(e: ft.DragStartEvent):
	index = layer_panel.get_layer_index_from_position(e.local_y)
	if index >= len(layer_panel.layers):
		return
	layer_panel.layer_being_moved = layer_panel.layers[index]
	layer_panel.make_layer_active(layer_panel.layer_being_moved.index)
	layer_panel.update()

def on_layer_drag(e: ft.DragUpdateEvent):
	if not layer_panel.layer_being_moved:
		return
	index = layer_panel.get_layer_index_from_position(e.local_y)
	if index == layer_panel.layer_being_moved.index:
		return
	layer_panel.move_layer(layer_panel.layer_being_moved, index)

def drop_layer(e: ft.DragEndEvent):
	e.page.refresh_layers()
	layer_panel.layer_being_moved = None


# LayerPanel == ft.Container
layer_panel = LayerPanel(
		content = ft.GestureDetector(
				content = ft.Column(
						controls = [],
						alignment = 'start',
						expand = True,
						spacing = 0,
						scroll = 'hidden',
				),
				drag_interval = 10,
				on_tap_down = layer_left_click,
				on_secondary_tap_down = layer_right_click,
				on_vertical_drag_start = pickup_layer,
				on_vertical_drag_update = on_layer_drag,
				on_vertical_drag_end = drop_layer,
		),
)

layer_panel.layers = []
layer_panel.layer_being_moved = None
layer_panel.layer_last_index = 0


# AssetPanel == ft.Container
asset_panel = AssetPanel(
		content = ft.Column(
				controls = [
					ft.Text("Under Construction"),
				],
		),
)

def resize_asset_manager(e):
	asset_manager.resize_asset_manager(e)

def realign_canvas(e):
	e.page.align_canvas()

asset_manager_dragbar = ft.GestureDetector(
		mouse_cursor = ft.MouseCursor.RESIZE_COLUMN,
		drag_interval = 50,
		on_pan_update = resize_asset_manager,
		on_pan_end = realign_canvas,
		content = ft.VerticalDivider(),
)


# AssetManager == ft.Container
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
asset_manager.layer_panel = layer_panel
asset_manager.asset_panel = asset_panel
asset_manager.dragbar = asset_manager_dragbar

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

# flet_layer_manager.py

# Flet imports
import flet as ft
from scripts import flet_utils


class LayerManager(ft.Container):
	# first make a column to hold the layers
	def make_layer_holder(self):
		layer_holder = ft.DragTarget(
			group = 'layer',
			content = ft.Column(
					spacing = 0,
					scroll = 'auto',
					controls = [],
			),
			on_will_accept = self.layer_will_accept,
			on_accept = self.layer_accept,
			on_leave = self.layer_leave,
		)
		return layer_holder

	# make a slot for each layer to go in
	def make_layer_slot(self):
		layer_slot = LayerSlot(
				group = 'layer',
				content = ft.Container(
						content = self.make_layer_display(),
				),
				on_will_accept = self.layer_slot_will_accept,
				on_accept = self.layer_slot_accept,
				on_leave = self.layer_slot_leave,
				data = {
						'index': -1,
						'type': 'slot',
						'has_spacer': False,
						'image': None,
				}
		)
		return layer_slot

	# make the display for the layer
	def make_layer_display(self):
		try:
			self.layer_count += 1
		except AttributeError:
			self.layer_count = 1

		layer_display = ft.Column(
				controls = [
						ft.Container(
								content = ft.Divider(
										height = 10,
										color = ft.colors.BLACK,
								),
								visible = False,
						),
						ft.Container(
								content = ft.Row(
										controls = [],

								),
								data = {
										'visible':True,
										'locked':False,
								},
								bgcolor = ft.colors.WHITE30,
								padding = 4,
						),
				],
				spacing = 0,
		)
		layer_show_hide = ft.IconButton(
				icon = ft.icons.VISIBILITY,
				tooltip = 'show/hide',
				on_click = self.show_hide_layer,
				data = {'parent':layer_display.controls[1]},
		)
		layer_lock_unlock = ft.IconButton(
				icon = ft.icons.LOCK_OPEN_OUTLINED,
				tooltip = 'lock/unlock',
				on_click = self.lock_unlock_layer,
				data = {'parent':layer_display.controls[1]},
		)
		layer_label = ft.TextField(
				value = ("layer_" + str(self.layer_count)),
				data = {'parent':layer_display.controls[1]},
				content_padding = 10,
				expand = True,
		)
		layer_handle = ft.GestureDetector(
				content = ft.Draggable(
						group = 'layer',
						content = ft.Icon(
								name = ft.icons.DRAG_HANDLE,
								data = {'parent':layer_display.controls[1]},
								tooltip = 'drag to move',
						),
				),
				on_secondary_tap = self.layer_right_click
		)
		layer_display.controls[1].content.controls.extend([layer_show_hide,layer_lock_unlock,layer_label,layer_handle])
		return layer_display

	# update all layer info
	def update_layers(self):
		self.page.layer_list = self.content.content.controls
		self.update_layer_indexes()
		self.update_visible_layer_list()
		self.update_active_layer_list()
		self.update()
		self.page.refresh_canvas()

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

	# handle layer shuffling
	def update_layer_indexes(self):
		layer_list = self.page.layer_list
		index = 0
		for layer in layer_list:
			if layer.data['type'] == 'slot':
				layer.data['index'] = index
				index += 1

	def move_layer_slot(self, index):
		layer_list = self.page.layer_list
		self.data['layer_being_moved'] = layer_list.pop(index)
		self.data['layer_last_index'] = index
		self.update_layers()

	def insert_layer_slot(self, index):
		layer_list = self.page.layer_list
		layer_list.insert(index,self.data['layer_being_moved'])
		self.data['layer_being_moved'] = None
		self.data['layer_last_index'] = -1
		self.update_layers()

	def layer_slot_will_accept(self, e):
		if not self.data['layer_being_moved']:
			return
		layer_list = self.page.layer_list
		index = e.control.data['index']
		e.control.show_layer_spacer()
		self.update_layers()

	def layer_slot_accept(self, e):
		if not self.data['layer_being_moved']:
			return
		layer_list = self.page.layer_list
		index = e.control.data['index']
		e.control.hide_layer_spacer()
		self.insert_layer_slot(index)

	def layer_slot_leave(self, e):
		layer_list = self.page.layer_list
		index = e.control.data['index']
		e.control.hide_layer_spacer()
		if self.data['layer_being_moved']:
			return
		self.move_layer_slot(index)

	# catch layers that are dropped outside slots
	def layer_will_accept(self, e):
		if not self.data['layer_being_moved']:
			return
		layer_list = self.page.layer_list
		if layer_list:
			if layer_list[-1].data['type'] != 'spacer':
				layer_list.append(
					ft.Container(
							content = ft.Divider(height = 10,color = ft.colors.BLACK),
							data = {'type':'spacer'}
					)
				)
		else:
			layer_list.append(
				ft.Container(
						content = ft.Divider(height = 10,color = ft.colors.BLACK),
						data = {'type':'spacer'}
				)
			)
		self.update_layers()

	def layer_accept(self, e):
		if not self.data['layer_being_moved']:
			return
		layer_list = self.page.layer_list
		if layer_list:
			if layer_list[-1].data['type'] == 'spacer':
				layer_list.pop(-1)
		layer_list.append(self.data['layer_being_moved'])
		self.data['layer_being_moved'] = None
		self.update_layers()

	def layer_leave(self, e):
		if not self.data['layer_being_moved']:
			return
		layer_list = self.page.layer_list
		if layer_list:
			if layer_list[-1].data['type'] == 'spacer':
				layer_list.pop(-1)
		self.update_layers()

	# handle toolbar functions
	def add_blank_layer(self, e):
		layer_list = self.page.layer_list
		layer_slot = self.make_layer_slot()
		layer_slot.data['image'] = flet_utils.create_blank_image()
		layer_list.append(layer_slot)
		self.page.message("added blank layer to canvas")
		self.update_layers()

	def add_images_as_layers(self, images):
		layer_list = self.page.layer_list
		for img in images:
			if images[img] == None:
				continue
			layer_slot = self.make_layer_slot()
			layer_slot.set_layer_slot_name(img)
			layer_slot.data['image'] = images[img]
			layer_list.append(layer_slot)
			self.page.message(f'added "{img}" as layer')
		self.update_layers()

	# fetch layer option on right click
	def layer_right_click(self,e):
		pass


# make each layer slot a target
class LayerSlot(ft.DragTarget):
	def set_layer_slot_name(self, name):
		self.content.content.controls[1].content.controls[1].value = name

	def show_layer_spacer(self):
		if not self.data['has_spacer']:
			self.data['has_spacer'] = True
			self.content.content.controls[0].visible = True
			self.update()

	def hide_layer_spacer(self):
		if self.data['has_spacer']:
			self.data['has_spacer'] = False
			self.content.content.controls[0].visible = False
			self.update()


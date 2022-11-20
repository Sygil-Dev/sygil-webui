## webui_utils.py


###### Textual Inversion #####################################################
textual_inversion_grid_row_list = [
	'model', 'medium', 'artist', 'trending', 'movement', 'flavors', 'techniques', 'tags',
	]

def get_textual_inversion_row_value(row_name):
	## lookup value
	pass

## wrapper functions
def load_blip_model():
	pass

def generate_caption():
	pass

def interrogate(image, models):
	result = {}
	load_blip_model()
	generate_caption()
	### magic ?????
	return result

def img2txt(data):
	## iterate through images
	for i in range(len(data['selected_images'])):
		result = interrogate(data['selected_images'][i], models = data['selected_models'])
		data['results'][i] = result

def run_textual_inversion(data):
	## reload model, pipe, upscalers

	## run clip interrogator
	img2txt(data)

## so far data(object) needs---> 	list of selected models
##									list of selected images
##		guessing functions also need some way of accessing settings...?
##		thinking only way to do that is to store everything in the generate button
##		and update on press. i'll stress test later and see if it slows things down
##		...adding data['results'] to pass data back

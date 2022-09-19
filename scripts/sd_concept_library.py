# base webui import and utils.
from webui_streamlit import st
from sd_utils import *

# streamlit imports


#other imports

# Temp imports 


# end of imports
#---------------------------------------------------------------------------------------------------------------

import os
from PIL import Image

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass

class plugin_info():
	plugname = "concept_library"
	description = "Concept Library"
	displayPriority = 4

def getLatestGeneratedImagesFromPath():
	#get the latest images from the generated images folder
	#get the path to the generated images folder
	generatedImagesPath = os.path.join(os.getcwd(), st.session_state['defaults'].general.sd_concepts_library_folder)
	#get all the files from the folders and subfolders
	files = []
	ext = ('jpeg', 'jpg', "png")
	#get the latest 10 images from the output folder without walking the subfolders
	for r, d, f in os.walk(generatedImagesPath):
		
		for file in f:
			if file.endswith(ext):
				files.append(os.path.join(r, file))
	#sort the files by date
	files.sort(reverse=True, key=os.path.getmtime)
	latest = files
	latest.reverse()

	# reverse the list so the latest images are first and truncate to
	# a reasonable number of images, 10 pages worth
	return [Image.open(f) for f in latest]

def layout():
	st.markdown(f"<h1 style='text-align: center; color: white;'>Navigate 300+ Textual-Inversion community trained concepts</h1>", unsafe_allow_html=True)
	
	latestImages = getLatestGeneratedImagesFromPath()
	st.session_state['latestImages'] = latestImages	
	
	#with history_tab:
	##---------------------------------------------------------
	## image slideshow test
	## Number of entries per screen
	#slideshow_N = 9
	#slideshow_page_number = 0	
	#slideshow_last_page = len(latestImages) // slideshow_N	
		
	## Add a next button and a previous button
	
	#slideshow_prev, slideshow_image_col , slideshow_next = st.columns([1, 10, 1])	
	
	#with slideshow_image_col:
		#slideshow_image = st.empty()
		
		#slideshow_image.image(st.session_state['latestImages'][0])
	
	#current_image = 0
	
	#if slideshow_next.button("Next", key=1):
		##print (current_image+1)
		#current_image = current_image+1
		#slideshow_image.image(st.session_state['latestImages'][current_image+1])
	#if slideshow_prev.button("Previous", key=0):
		##print ([current_image-1])
		#current_image = current_image-1
		#slideshow_image.image(st.session_state['latestImages'][current_image - 1])


	#---------------------------------------------------------

	# image gallery
	# Number of entries per screen
	gallery_N = 9
	if not "galleryPage" in st.session_state:
		st.session_state["galleryPage"] = 0
	gallery_last_page = len(latestImages) // gallery_N

	# Add a next button and a previous button

	gallery_prev, gallery_refresh, gallery_pagination , gallery_next = st.columns([2, 2, 8, 1])

	# the pagination doesnt work for now so its better to disable the buttons.

	if gallery_refresh.button("Refresh", key=4):
		st.session_state["galleryPage"] = 0

	if gallery_next.button("Next", key=3):

		if st.session_state["galleryPage"] + 1 > gallery_last_page:
			st.session_state["galleryPage"] = 0
		else:
			st.session_state["galleryPage"] += 1

	if gallery_prev.button("Previous", key=2):

		if st.session_state["galleryPage"] - 1 < 0:
			st.session_state["galleryPage"] = gallery_last_page
		else:
			st.session_state["galleryPage"] -= 1

	#print(st.session_state["galleryPage"])
	# Get start and end indices of the next page of the dataframe
	gallery_start_idx = st.session_state["galleryPage"] * gallery_N
	gallery_end_idx = (1 + st.session_state["galleryPage"]) * gallery_N

	#---------------------------------------------------------

	placeholder = st.empty()

	#populate the 3 images per column
	with placeholder.container():
		col1, col2, col3 = st.columns(3)
		col1_cont = st.container()
		col2_cont = st.container()
		col3_cont = st.container()

		#print (len(st.session_state['latestImages']))
		images = list(reversed(st.session_state['latestImages']))[gallery_start_idx:(gallery_start_idx+gallery_N)]

		with col1_cont:
			with col1:
				[st.image(images[index], caption="") for index in [0, 3, 6] if index < len(images)]
		with col2_cont:
			with col2:
				[st.image(images[index]) for index in [1, 4, 7] if index < len(images)]
		with col3_cont:
			with col3:
				[st.image(images[index]) for index in [2, 5, 8] if index < len(images)]
	

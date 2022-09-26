# This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

# Copyright 2022 sd-webui team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
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
	plugname = "home"
	description = "Home"
	isTab = True
	displayPriority = 0

def getLatestGeneratedImagesFromPath():
	#get the latest images from the generated images folder
	#get the path to the generated images folder
	generatedImagesPath = os.path.join(os.getcwd(),'outputs')
	#get all the files from the folders and subfolders
	files = []
	#get the latest 10 images from the output folder without walking the subfolders
	for r, d, f in os.walk(generatedImagesPath):
		for file in f:
			if '.png' in file:
				files.append(os.path.join(r, file))
	#sort the files by date
	files.sort(reverse=True, key=os.path.getmtime)
	latest = files[:90]
	latest.reverse()

	# reverse the list so the latest images are first and truncate to
	# a reasonable number of images, 10 pages worth
	return [Image.open(f) for f in latest]

def get_images_from_lexica():
	#scrape images from lexica.art
	#get the html from the page
	#get the html with cookies and javascript
	apiEndpoint = r'https://lexica.art/api/trpc/prompts.infinitePrompts?batch=1&input=%7B%220%22%3A%7B%22json%22%3A%7B%22limit%22%3A10%2C%22text%22%3A%22%22%2C%22cursor%22%3A10%7D%7D%7D'
	#REST API call
	# 
	from requests_html import HTMLSession
	session = HTMLSession()

	response = session.get(apiEndpoint)
	#req = requests.Session()
	#req.headers['user-agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
	#response = req.get(apiEndpoint)
	print(response.status_code)
	print(response.text)
	#get the json from the response
	#json = response.json()
	#get the prompts from the json
	print(response)
	#session = requests.Session()
	#parseEndpointJson = session.get(apiEndpoint,headers=headers,verify=False)
	#print(parseEndpointJson)
	#print('test2')
	#page = requests.get("https://lexica.art/", headers={'User-Agent': 'Mozilla/5.0'})
	#parse the html
	#soup = BeautifulSoup(page.content, 'html.parser')
	#find all the images
	#print(soup)
	#images = soup.find_all('alt-image')
	#create a list to store the image urls
	image_urls = []
	#loop through the images
	for image in images:
		#get the url
		image_url = image['src']
		#add it to the list
		image_urls.append('http://www.lexica.art/'+image_url)
	#return the list
	print(image_urls)
	return image_urls

def layout():
	#streamlit home page layout
	#center the title
	st.markdown("<h1 style='text-align: center; color: white;'>Welcome, let's make some 🎨</h1>", unsafe_allow_html=True)
	#make a gallery of images
	#st.markdown("<h2 style='text-align: center; color: white;'>Gallery</h2>", unsafe_allow_html=True)
	#create a gallery of images using columns
	#col1, col2, col3 = st.columns(3)
	#load the images
	#create 3 columns
	# create a tab for the gallery
	#st.markdown("<h2 style='text-align: center; color: white;'>Gallery</h2>", unsafe_allow_html=True)
	#st.markdown("<h2 style='text-align: center; color: white;'>Gallery</h2>", unsafe_allow_html=True)
	
	history_tab, discover_tabs = st.tabs(["History","Discover"])
	
	latestImages = getLatestGeneratedImagesFromPath()
	st.session_state['latestImages'] = latestImages	
	
	with history_tab:
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

		print(st.session_state["galleryPage"])
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

			print (len(st.session_state['latestImages']))
			images = list(reversed(st.session_state['latestImages']))[gallery_start_idx:(gallery_start_idx+gallery_N)]

			with col1_cont:
				with col1:
					[st.image(images[index]) for index in [0, 3, 6] if index < len(images)]
			with col2_cont:
				with col2:
					[st.image(images[index]) for index in [1, 4, 7] if index < len(images)]
			with col3_cont:
				with col3:
					[st.image(images[index]) for index in [2, 5, 8] if index < len(images)]
							
					
		st.session_state['historyTab'] = [history_tab,col1,col2,col3,placeholder,col1_cont,col2_cont,col3_cont]		

	with discover_tabs:
		st.markdown("<h1 style='text-align: center; color: white;'>Soon :)</h1>", unsafe_allow_html=True)
	
	#display the images
	#add a button to the gallery
	#st.markdown("<h2 style='text-align: center; color: white;'>Try it out</h2>", unsafe_allow_html=True)
	#create a button to the gallery
	#if st.button("Try it out"):
		#if the button is clicked, go to the gallery
		#st.experimental_rerun()

# This file is part of sygil-webui (https://github.com/Sygil-Dev/sygil-webui/).

# Copyright 2022 Sygil-Dev team.
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
from sd_utils import st

# streamlit imports
import streamlit.components.v1 as components
#other imports

import os, math
from PIL import Image

# Temp imports
#from basicsr.utils.registry import ARCH_REGISTRY


# end of imports
#---------------------------------------------------------------------------------------------------------------

# Init Vuejs component
_component_func = components.declare_component(
	"sd-concepts-browser", "./frontend/dists/concept-browser/dist")


def sdConceptsBrowser(concepts, key=None):
	component_value = _component_func(concepts=concepts, key=key, default="")
	return component_value


@st.experimental_memo(persist="disk", show_spinner=False, suppress_st_warning=True)
def getConceptsFromPath(page, conceptPerPage, searchText=""):
	#print("getConceptsFromPath", "page:", page, "conceptPerPage:", conceptPerPage, "searchText:", searchText)
	# get the path where the concepts are stored
	path = os.path.join(
		os.getcwd(),  st.session_state['defaults'].general.sd_concepts_library_folder)
	acceptedExtensions = ('jpeg', 'jpg', "png")
	concepts = []

	if os.path.exists(path):
		# List all folders (concepts) in the path
		folders = [f for f in os.listdir(
			path) if os.path.isdir(os.path.join(path, f))]
		filteredFolders = folders

		# Filter the folders by the search text
		if searchText != "":
			filteredFolders = [
				f for f in folders if searchText.lower() in f.lower()]
	else:
		filteredFolders = []

	conceptIndex = 1
	for folder in filteredFolders:
		# handle pagination
		if conceptIndex > (page * conceptPerPage):
			continue
		if conceptIndex <= ((page - 1) * conceptPerPage):
			conceptIndex += 1
			continue

		concept = {
			"name": folder,
			"token": "<" + folder + ">",
			"images": [],
			"type": ""
		}

		# type of concept is inside type_of_concept.txt
		typePath = os.path.join(path, folder, "type_of_concept.txt")
		binFile = os.path.join(path, folder, "learned_embeds.bin")

		# Continue if the concept is not valid or the download has failed (no type_of_concept.txt or no binFile)
		if not os.path.exists(typePath) or not os.path.exists(binFile):
			continue

		with open(typePath, "r") as f:
			concept["type"] = f.read()

		# List all files in the concept/concept_images folder
		files = [f for f in os.listdir(os.path.join(path, folder, "concept_images")) if os.path.isfile(
			os.path.join(path, folder, "concept_images", f))]
		# Retrieve only the 4 first images
		for file in files:

			# Skip if we already have 4 images
			if len(concept["images"]) >= 4:
				break

			if file.endswith(acceptedExtensions):
				try:
					# Add a copy of the image to avoid file locking
					originalImage = Image.open(os.path.join(
						path, folder, "concept_images", file))

					# Maintain the aspect ratio (max 200x200)
					resizedImage = originalImage.copy()
					resizedImage.thumbnail((200, 200), Image.ANTIALIAS)

					# concept["images"].append(resizedImage)

					concept["images"].append(imageToBase64(resizedImage))
					# Close original image
					originalImage.close()
				except:
					print("Error while loading image", file, "in concept", folder, "(The file may be corrupted). Skipping it.")

		concepts.append(concept)
		conceptIndex += 1
	# print all concepts name
	#print("Results:", [c["name"] for c in concepts])
	return concepts

@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def imageToBase64(image):
	import io
	import base64
	buffered = io.BytesIO()
	image.save(buffered, format="PNG")
	img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
	return img_str


@st.experimental_memo(persist="disk", show_spinner=False, suppress_st_warning=True)
def getTotalNumberOfConcepts(searchText=""):
	# get the path where the concepts are stored
	path = os.path.join(
		os.getcwd(),  st.session_state['defaults'].general.sd_concepts_library_folder)
	concepts = []

	if os.path.exists(path):
		# List all folders (concepts) in the path
		folders = [f for f in os.listdir(
			path) if os.path.isdir(os.path.join(path, f))]
		filteredFolders = folders

		# Filter the folders by the search text
		if searchText != "":
			filteredFolders = [
				f for f in folders if searchText.lower() in f.lower()]
	else:
		filteredFolders = []
	return len(filteredFolders)



def layout():
	# 2 tabs, one for Concept Library and one for the Download Manager
	tab_library, tab_downloader = st.tabs(["Library", "Download Manager"])

	# Concept Library
	with tab_library:
		downloaded_concepts_count = getTotalNumberOfConcepts()
		concepts_per_page = st.session_state["defaults"].concepts_library.concepts_per_page

		if not "results" in st.session_state:
			st.session_state["results"] = getConceptsFromPath(1, concepts_per_page, "")

		# Pagination controls
		if not "cl_current_page" in st.session_state:
			st.session_state["cl_current_page"] = 1

		# Search
		if not 'cl_search_text' in st.session_state:
			st.session_state["cl_search_text"] = ""

		if not 'cl_search_results_count' in st.session_state:
			st.session_state["cl_search_results_count"] = downloaded_concepts_count

		# Search bar
		_search_col, _refresh_col = st.columns([10, 2])
		with _search_col:
			search_text_input = st.text_input("Search", "", placeholder=f'Search for a concept ({downloaded_concepts_count} available)', label_visibility="hidden")
			if search_text_input != st.session_state["cl_search_text"]:
				# Search text has changed
				st.session_state["cl_search_text"] = search_text_input
				st.session_state["cl_current_page"] = 1
				st.session_state["cl_search_results_count"] = getTotalNumberOfConcepts(st.session_state["cl_search_text"])
				st.session_state["results"] = getConceptsFromPath(1, concepts_per_page, st.session_state["cl_search_text"])

		with _refresh_col:
			# Super weird fix to align the refresh button with the search bar ( Please streamlit, add css support..  )
			_refresh_col.write("")
			_refresh_col.write("")
			if st.button("Refresh concepts", key="refresh_concepts", help="Refresh the concepts folders. Use this if you have added new concepts manually or deleted some."):
				getTotalNumberOfConcepts.clear()
				getConceptsFromPath.clear()
				st.experimental_rerun()


		# Show results
		results_empty = st.empty()

		# Pagination
		pagination_empty = st.empty()

		# Layouts
		with pagination_empty:
			with st.container():
				if len(st.session_state["results"]) > 0:
					last_page = math.ceil(st.session_state["cl_search_results_count"] / concepts_per_page)
					_1, _2, _3, _4, _previous_page, _current_page, _next_page, _9, _10, _11, _12 = st.columns([1,1,1,1,1,2,1,1,1,1,1])

					# Previous page
					with _previous_page:
						if st.button("Previous", key="cl_previous_page"):
							st.session_state["cl_current_page"] -= 1
							if st.session_state["cl_current_page"] <= 0:
								st.session_state["cl_current_page"] = last_page
							st.session_state["results"] = getConceptsFromPath(st.session_state["cl_current_page"], concepts_per_page, st.session_state["cl_search_text"])

					# Current page
					with _current_page:
						_current_page_container = st.empty()

					# Next page
					with _next_page:
						if st.button("Next", key="cl_next_page"):
							st.session_state["cl_current_page"] += 1
							if st.session_state["cl_current_page"] > last_page:
								st.session_state["cl_current_page"] = 1
							st.session_state["results"] = getConceptsFromPath(st.session_state["cl_current_page"], concepts_per_page, st.session_state["cl_search_text"])

					# Current page
					with _current_page_container:
						st.markdown(f'<p style="text-align: center">Page {st.session_state["cl_current_page"]} of {last_page}</p>', unsafe_allow_html=True)
						# st.write(f"Page {st.session_state['cl_current_page']} of {last_page}", key="cl_current_page")

		with results_empty:
			with st.container():
				if downloaded_concepts_count == 0:
					st.write("You don't have any concepts in your library ")
					st.markdown("To add concepts to your library, download some from the [sd-concepts-library](https://github.com/Sygil-Dev/sd-concepts-library) \
						repository and save the content of `sd-concepts-library` into ```./models/custom/sd-concepts-library``` or just create your own concepts :wink:.", unsafe_allow_html=False)
				else:
					if len(st.session_state["results"]) == 0:
						st.write("No concept found in the library matching your search: " + st.session_state["cl_search_text"])
					else:
						# display number of results
						if st.session_state["cl_search_text"]:
							st.write(f"Found {st.session_state['cl_search_results_count']} {'concepts' if st.session_state['cl_search_results_count'] > 1 else 'concept' } matching your search")
						sdConceptsBrowser(st.session_state['results'], key="results")


	with tab_downloader:
		st.write("Not implemented yet")

	return False

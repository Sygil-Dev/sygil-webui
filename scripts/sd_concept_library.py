# base webui import and utils.
from webui_streamlit import st
from sd_utils import *

# streamlit imports
import streamlit.components.v1 as components

#other imports

#from sd_concept_browser import *

# Temp imports


# end of imports
#---------------------------------------------------------------------------------------------------------------

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass

# parent_dir = os.path.dirname(os.path.abspath(__file__))
# build_dir = os.path.join(parent_dir, "frontend/dist")
_component_func = components.declare_component("sd-concepts-browser", "./frontend/dist")

class plugin_info():
	plugname = "concept_library"
	description = "Concept Library"
	displayPriority = 4

def sdConceptsBrowser(concepts, key=None):
	component_value = _component_func(concepts=concepts, key=key, default="")
	return component_value

@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def getConceptsFromPath(page, conceptPerPage, searchText= ""):
	#print("getConceptsFromPath", "page:", page, "conceptPerPage:", conceptPerPage, "searchText:", searchText)
	# get the path where the concepts are stored
	path = os.path.join(os.getcwd(),  st.session_state['defaults'].general.sd_concepts_library_folder)
	acceptedExtensions = ('jpeg', 'jpg', "png")
	concepts = []

	if os.path.exists(path):
		# List all folders (concepts) in the path
		folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
		filteredFolders = folders

		# Filter the folders by the search text
		if searchText != "":
			filteredFolders = [f for f in folders if searchText.lower() in f.lower()]
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
		files = [f for f in os.listdir(os.path.join(path, folder, "concept_images")) if os.path.isfile(os.path.join(path, folder, "concept_images", f))]
		# Retrieve only the 4 first images
		for file in files[:4]:
			if file.endswith(acceptedExtensions):
				# Add a copy of the image to avoid file locking
				originalImage = Image.open(os.path.join(path, folder, "concept_images", file))

				# Maintain the aspect ratio (max 200x200)
				resizedImage = originalImage.copy()
				resizedImage.thumbnail((200, 200), Image.ANTIALIAS)

				#concept["images"].append(resizedImage)

				concept["images"].append(imageToBase64(resizedImage))
				# Close original image
				originalImage.close()

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

@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def getTotalNumberOfConcepts(searchText= ""):
	# get the path where the concepts are stored
	path = os.path.join(os.getcwd(),  st.session_state['defaults'].general.sd_concepts_library_folder)
	concepts = []

	if os.path.exists(path):
		# List all folders (concepts) in the path
		folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
		filteredFolders = folders

		# Filter the folders by the search text
		if searchText != "":
			filteredFolders = [f for f in folders if searchText.lower() in f.lower()]
	else:
		filteredFolders = []
	return len(filteredFolders)

def layout():
	# Pagination
	page = 1
	conceptPerPage = 12
	totalNumberOfConcepts = getTotalNumberOfConcepts()
	if not "cl_page" in st.session_state:
		st.session_state["cl_page"] = page
	if not "cl_conceptPerPage" in st.session_state:
		st.session_state["cl_conceptPerPage"] = conceptPerPage
	#Search for a concept (totalNumberOfConcepts available)
	searchInput = st.text_input("","", placeholder= f'Search for a concept ({totalNumberOfConcepts} available)')
	if searchInput != "":
		st.session_state["cl_page"] = 1
		totalNumberOfConcepts = getTotalNumberOfConcepts(searchInput)

	# Pagination
	last_page = math.ceil(getTotalNumberOfConcepts(searchInput) / st.session_state["cl_conceptPerPage"])
	_prev, _per_page ,_next = st.columns([1, 10, 1])
	if ("concepts" in st.session_state and len(st.session_state['concepts']) > 0):
		# The condition doesnt work, it should be fixed

		with _prev:
			if st.button("Previous", disabled = st.session_state["cl_page"] == 1):
				st.session_state["cl_page"] -= 1
				st.session_state['concepts'] = getConceptsFromPath(st.session_state["cl_page"], st.session_state["cl_conceptPerPage"], searchInput)

		with _per_page:
			st.caption("Page " + str(st.session_state["cl_page"]) + " / " + str(last_page))

		with _next:
			if st.button("Next", disabled = st.session_state["cl_page"] == last_page):
				st.session_state["cl_page"] += 1
				st.session_state['concepts'] = getConceptsFromPath(st.session_state["cl_page"], st.session_state["cl_conceptPerPage"], searchInput)

	placeholder = st.empty()

	with placeholder.container():
		# Init session state
		if not "concepts" in st.session_state:
			st.session_state['concepts'] = []

		# Refresh concepts
		st.session_state['concepts'] = getConceptsFromPath(st.session_state["cl_page"], st.session_state["cl_conceptPerPage"], searchInput)
		conceptsLenght = len(st.session_state['concepts'])

		if (conceptsLenght == 0):
			if (searchInput == ""):
				st.write("You don't have any concepts in your library ")
				# Propose the user to go to "https://github.com/sd-webui/sd-concepts-library"
				st.markdown("To add concepts to your library, download some from the [sd-concepts-library](https://github.com/sd-webui/sd-concepts-library) \
				repository and save the content of `sd-concepts-library` into ```./models/custom/sd-concepts-library``` or just create your own concepts :wink:.", unsafe_allow_html=False)
			else:
				st.write("No concepts found in the library matching your search: " + searchInput)

		# print("Number of concept matching the query:", conceptsLenght)
		sdConceptsBrowser(st.session_state['concepts'], key="clipboard")


	return False

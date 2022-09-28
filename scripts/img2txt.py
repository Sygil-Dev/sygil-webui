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

"""
CLIP Interrogator made by @pharmapsychotic modified to work with our WebUI.

# CLIP Interrogator by @pharmapsychotic 
Twitter: https://twitter.com/pharmapsychotic
Github: https://github.com/pharmapsychotic/clip-interrogator

Description:
What do the different OpenAI CLIP models see in an image? What might be a good text prompt to create similar images using CLIP guided diffusion
or another text to image model? The CLIP Interrogator is here to get you answers!

Please consider buying him a coffee via [ko-fi](https://ko-fi.com/pharmapsychotic) or following him on [twitter](https://twitter.com/pharmapsychotic).

And if you're looking for more Ai art tools check out my [Ai generative art tools list](https://pharmapsychotic.com/tools.html).

"""

#
# base webui import and utils.
from sd_utils import *

# streamlit imports
import streamlit_nested_layout

#streamlit components section
from streamlit_server_state import server_state, server_state_lock

#other imports
import hydralit_components as hc


# end of imports
#---------------------------------------------------------------------------------------------------------------

#
def layout():
	#set_page_title("Image-to-Text - Stable Diffusion WebUI")
	st.info("Under Construction. :construction_worker:")
	
	#theme_neutral = {'bgcolor': '#f9f9f9','title_color': 'black','content_color': 'black','icon_color': 'orange', 'icon': 'fa fa-question-circle'}
	#hc.info_card(title='Some heading GOOD', content='All good!', sentiment='good',bar_value=77)
	
	#hc.nav_bar([{'icon': "far fa-copy", 'label':"Left End"}, {'id':'Copy','icon':"🐙",'label':"Copy"},
				#{'icon': "fa-solid fa-radar",'label':"Dropdown1",
				 #' submenu':[{'id':' subid11','icon': "fa fa-paperclip", 'label':"Sub-item 1"},
							 #{'id':'subid12','icon': "💀", 'label':"Sub-item 2"},
							 #{'id':'subid13','icon': "fa fa-database", 'label':"Sub-item 3"}]}],
			   #override_theme=theme_neutral, hide_streamlit_markers=False)
	
	
	#with st.form("img2txt-inputs"):
		#st.session_state["generation_mode"] = "txt2img"

		#input_col1, generate_col1 = st.columns([10,1])

		#with input_col1:
			##prompt = st.text_area("Input Text","")
			#prompt = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")

		## Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
		#generate_col1.write("")
		#generate_col1.write("")
		#generate_button = generate_col1.form_submit_button("Generate")

		## creating the page layout using columns
		#col1, col2, col3 = st.columns([1,2,1], gap="large")   	
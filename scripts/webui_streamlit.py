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
#import streamlit as st

# We import hydralit like this to replace the previous stuff
# we had with native streamlit as it lets ur replace things 1:1
#import hydralit as st 
from sd_utils import *

# streamlit imports
import streamlit_nested_layout

#streamlit components section
from st_on_hover_tabs import on_hover_tabs
from streamlit_server_state import server_state, server_state_lock

#other imports

import warnings
import os, toml
import k_diffusion as K
from omegaconf import OmegaConf

if not "defaults" in st.session_state:
    st.session_state["defaults"] = {}
    
st.session_state["defaults"] = OmegaConf.load("configs/webui/webui_streamlit.yaml")

if (os.path.exists("configs/webui/userconfig_streamlit.yaml")):
    user_defaults = OmegaConf.load("configs/webui/userconfig_streamlit.yaml")
    st.session_state["defaults"] = OmegaConf.merge(st.session_state["defaults"], user_defaults)
else:
	OmegaConf.save(config=st.session_state.defaults, f="configs/webui/userconfig_streamlit.yaml")
	loaded = OmegaConf.load("configs/webui/userconfig_streamlit.yaml")
	assert st.session_state.defaults == loaded		

if (os.path.exists(".streamlit/config.toml")):
	st.session_state["streamlit_config"] = toml.load(".streamlit/config.toml")

# end of imports
#---------------------------------------------------------------------------------------------------------------


try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass

# remove some annoying deprecation warnings that show every now and then.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)     

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = str(st.session_state["defaults"].general.gpu)


# functions to load css locally OR remotely starts here. Options exist for future flexibility. Called as st.markdown with unsafe_allow_html as css injection
# TODO, maybe look into async loading the file especially for remote fetching 
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
	st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def load_css(isLocal, nameOrURL):
	if(isLocal):
		local_css(nameOrURL)
	else:
		remote_css(nameOrURL)

def layout():
	"""Layout functions to define all the streamlit layout here."""
	st.set_page_config(page_title="Stable Diffusion Playground", layout="wide")
	#app = st.HydraApp(title='Stable Diffusion WebUI', favicon="", sidebar_state="expanded",
	                  #hide_streamlit_markers=False, allow_url_nav=True , clear_cross_app_sessions=False)

	with st.empty():
		# load css as an external file, function has an option to local or remote url. Potential use when running from cloud infra that might not have access to local path.
		load_css(True, 'frontend/css/streamlit.main.css')
		
	# check if the models exist on their respective folders
	with server_state_lock["GFPGAN_available"]:
		if os.path.exists(os.path.join(st.session_state["defaults"].general.GFPGAN_dir, f"{st.session_state['defaults'].general.GFPGAN_model}.pth")):
			server_state["GFPGAN_available"] = True
		else:
			server_state["GFPGAN_available"] = False

	with server_state_lock["RealESRGAN_available"]:
		if os.path.exists(os.path.join(st.session_state["defaults"].general.RealESRGAN_dir, f"{st.session_state['defaults'].general.RealESRGAN_model}.pth")):
			server_state["RealESRGAN_available"] = True 
		else:
			server_state["RealESRGAN_available"] = False	

	with st.sidebar:		
		tabs = on_hover_tabs(tabName=['Stable Diffusion', "Textual Inversion","Model Manager","Settings"], 
                        iconName=['dashboard','model_training' ,'cloud_download', 'settings'], default_choice=0)
		
		# need to see how to get the icons to show for the hydralit option_bar				
		#tabs = hc.option_bar([{'icon':'grid-outline','label':'Stable Diffusion'}, {'label':"Textual Inversion"},
		                      #{'label':"Model Manager"},{'label':"Settings"}],
		                     #horizontal_orientation=False,
		                     #override_theme={'txc_inactive': 'white','menu_background':'#111', 'stVerticalBlock': '#111','txc_active':'yellow','option_active':'blue'})
		
	if tabs =='Stable Diffusion':
		# set the page url and title
		st.experimental_set_query_params(page='stable-diffusion')
		try:
			set_page_title("Stable Diffusion Playground")
		except NameError:
			st.experimental_rerun()
		
		txt2img_tab, img2img_tab, txt2vid_tab, img2txt_tab, concept_library_tab = st.tabs(["Text-to-Image", "Image-to-Image", 
		                                                                      "Text-to-Video", "Image-To-Text",
		                                                                      "Concept Library"])
		#with home_tab:
			#from home import layout
			#layout()		
		
		with txt2img_tab:
			from txt2img import layout
			layout()
		
		with img2img_tab:
			from img2img import layout
			layout()
		
		with txt2vid_tab:
			from txt2vid import layout
			layout()
			
		with img2txt_tab:
			from img2txt import layout
			layout()	
			
		with concept_library_tab:
			from sd_concept_library import layout
			layout()			
		
	#
	elif tabs == 'Model Manager':
		set_page_title("Model Manager - Stable Diffusion Playground")
		
		from ModelManager import layout
		layout()
	
	elif tabs == 'Textual Inversion':		
		from textual_inversion import layout
		layout()
		
	elif tabs == 'Settings':
		set_page_title("Settings - Stable Diffusion Playground")
		
		from Settings import layout
		layout()
	
if __name__ == '__main__':
	layout()     
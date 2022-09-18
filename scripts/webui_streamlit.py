# base webui import and utils.
import streamlit as st

# streamlit imports
import streamlit_nested_layout

#streamlit components section
from st_on_hover_tabs import on_hover_tabs

#other imports

import warnings
import os
import k_diffusion as K
from omegaconf import OmegaConf

from sd_utils import *
if not "defaults" in st.session_state:
    st.session_state["defaults"] = {}
    
st.session_state["defaults"] = OmegaConf.load("configs/webui/webui_streamlit.yaml")

if (os.path.exists("configs/webui/userconfig_streamlit.yaml")):
    user_defaults = OmegaConf.load("configs/webui/userconfig_streamlit.yaml")
    st.session_state["defaults"] = OmegaConf.merge(st.session_state["defaults"], user_defaults)

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

	with st.empty():
		# load css as an external file, function has an option to local or remote url. Potential use when running from cloud infra that might not have access to local path.
		load_css(True, 'frontend/css/streamlit.main.css')
		
	# check if the models exist on their respective folders
	if os.path.exists(os.path.join(st.session_state["defaults"].general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
		st.session_state["GFPGAN_available"] = True
	else:
		st.session_state["GFPGAN_available"] = False

	if os.path.exists(os.path.join(st.session_state["defaults"].general.RealESRGAN_dir, "experiments","pretrained_models", f"{st.session_state['defaults'].general.RealESRGAN_model}.pth")):
		st.session_state["RealESRGAN_available"] = True
	else:
		st.session_state["RealESRGAN_available"] = False	
		
	# Allow for custom models to be used instead of the default one,
	# an example would be Waifu-Diffusion or any other fine tune of stable diffusion
	st.session_state["custom_models"]:sorted = []
	for root, dirs, files in os.walk(os.path.join("models", "custom")):
		for file in files:
			if os.path.splitext(file)[1] == '.ckpt':
				#fullpath = os.path.join(root, file)
				#print(fullpath)
				st.session_state["custom_models"].append(os.path.splitext(file)[0])
				#print (os.path.splitext(file)[0])
	
	if len(st.session_state["custom_models"]) > 0:
		st.session_state["CustomModel_available"] = True
		st.session_state["custom_models"].append("Stable Diffusion v1.4")
	else:
		st.session_state["CustomModel_available"] = False

	with st.sidebar:
		# The global settings section will be moved to the Settings page.
		#with st.expander("Global Settings:"):
		#st.write("Global Settings:")
		#defaults.general.update_preview = st.checkbox("Update Image Preview", value=defaults.general.update_preview,
                                                              #help="If enabled the image preview will be updated during the generation instead of at the end. You can use the Update Preview \
							      #Frequency option bellow to customize how frequent it's updated. By default this is enabled and the frequency is set to 1 step.")
		#st.session_state.update_preview_frequency = st.text_input("Update Image Preview Frequency", value=defaults.general.update_preview_frequency,
                                                                          #help="Frequency in steps at which the the preview image is updated. By default the frequency is set to 1 step.")
		
		tabs = on_hover_tabs(tabName=['Stable Diffusion', "Textual Inversion","Model Manager","Settings"], 
                         iconName=['dashboard','model_training' ,'cloud_download', 'settings'], default_choice=0)
		
	if tabs =='Stable Diffusion':		
		# txt2img_tab, img2img_tab, txt2vid_tab, postprocessing_tab, concept_library_tab = st.tabs(["Text-to-Image Unified", "Image-to-Image Unified", 
	    #                                                                                             "Text-to-Video","Post-Processing", "Concept Library"])
		txt2img_tab, img2img_tab, txt2vid_tab = st.tabs(
			["Text-to-Image Unified", "Image-to-Image Unified", "Text-to-Video"]
		)
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
			
		# with concept_library_tab:
		# 	from sd_concept_library import layout
		# 	layout()			
		
	#
	elif tabs == 'Model Manager':
		from ModelManager import layout
		layout()
	
	# elif tabs == 'Textual Inversion':
	# 	from textual_inversion import layout
	# 	layout()	
	
if __name__ == '__main__':
	layout()     
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
import argparse


# end of imports
#---------------------------------------------------------------------------------------------------------------

load_configs()

help = """
A double dash (`--`) is used to separate streamlit arguments from app arguments.
As a result using "streamlit run webui_streamlit.py --headless"
will show the help for streamlit itself and not pass any argument to our app,
we need to use "streamlit run webui_streamlit.py -- --headless"
in order to pass a command argument to this app."""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--headless", action='store_true', help="Don't launch web server, util if you just want to run the stable horde bridge.", default=False)

parser.add_argument("--bridge", action='store_true', help="don't launch web server, but make this instance into a Horde bridge.", default=False)
parser.add_argument('--horde_api_key', action="store", required=False, type=str, help="The API key corresponding to the owner of this Horde instance")
parser.add_argument('--horde_name', action="store", required=False, type=str, help="The server name for the Horde. It will be shown to the world and there can be only one.")
parser.add_argument('--horde_url', action="store", required=False, type=str, help="The SH Horde URL. Where the bridge will pickup prompts and send the finished generations.")
parser.add_argument('--horde_priority_usernames',type=str, action='append', required=False, help="Usernames which get priority use in this horde instance. The owner's username is always in this list.")
parser.add_argument('--horde_max_power',type=int, required=False, help="How much power this instance has to generate pictures. Min: 2")
parser.add_argument('--horde_sfw', action='store_true', required=False, help="Set to true if you do not want this worker generating NSFW images.")
parser.add_argument('--horde_blacklist', nargs='+', required=False, help="List the words that you want to blacklist.")
parser.add_argument('--horde_censorlist', nargs='+', required=False, help="List the words that you want to censor.")
parser.add_argument('--horde_censor_nsfw', action='store_true', required=False, help="Set to true if you want this bridge worker to censor NSFW images.")
parser.add_argument('--horde_model', action='store', required=False, help="Which model to run on this horde.")
parser.add_argument('-v', '--verbosity', action='count', default=0, help="The default logging level is ERROR or higher. This value increases the amount of logging seen in your screen")
parser.add_argument('-q', '--quiet', action='count', default=0, help="The default logging level is ERROR or higher. This value decreases the amount of logging seen in your screen")
opt = parser.parse_args()

with server_state_lock["bridge"]:
    server_state["bridge"] = opt.bridge

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

@logger.catch(reraise=True)
def layout():
        """Layout functions to define all the streamlit layout here."""
        if not st.session_state["defaults"].debug.enable_hydralit:
            st.set_page_config(page_title="Stable Diffusion Playground", layout="wide")

        #app = st.HydraApp(title='Stable Diffusion WebUI', favicon="", sidebar_state="expanded", layout="wide",
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

            #with inpainting_tab:
                #from inpainting import layout
                #layout()

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
    set_logger_verbosity(opt.verbosity)
    quiesce_logger(opt.quiet)

    if not opt.headless:
        layout()

    with server_state_lock["bridge"]:
        if server_state["bridge"]:
            try:
                import bridgeData as cd
            except ModuleNotFoundError as e:
                logger.warning("No bridgeData found. Falling back to default where no CLI args are set.")
                logger.debug(str(e))
            except SyntaxError as e:
                logger.warning("bridgeData found, but is malformed. Falling back to default where no CLI args are set.")
                logger.debug(str(e))
            except Exception as e:
                logger.warning("No bridgeData found, use default where no CLI args are set")
                logger.debug(str(e))
            finally:
                try: # check if cd exists (i.e. bridgeData loaded properly)
                    cd
                except: # if not, create defaults
                    class temp(object):
                        def __init__(self):
                            random.seed()
                            self.horde_url = "https://stablehorde.net"
                            # Give a cool name to your instance
                            self.horde_name = f"Automated Instance #{random.randint(-100000000, 100000000)}"
                            # The api_key identifies a unique user in the horde
                            self.horde_api_key = "0000000000"
                            # Put other users whose prompts you want to prioritize.
                            # The owner's username is always included so you don't need to add it here, unless you want it to have lower priority than another user
                            self.horde_priority_usernames = []
                            self.horde_max_power = 8
                            self.nsfw = True
                            self.censor_nsfw = False
                            self.blacklist = []
                            self.censorlist = []
                            self.models_to_load = ["stable_diffusion"]
                    cd = temp()
            horde_api_key = opt.horde_api_key if opt.horde_api_key else cd.horde_api_key
            horde_name = opt.horde_name if opt.horde_name else cd.horde_name
            horde_url = opt.horde_url if opt.horde_url else cd.horde_url
            horde_priority_usernames = opt.horde_priority_usernames if opt.horde_priority_usernames else cd.horde_priority_usernames
            horde_max_power = opt.horde_max_power if opt.horde_max_power else cd.horde_max_power
            # Not used yet
            horde_models = [opt.horde_model] if opt.horde_model else cd.models_to_load
            try:
                horde_nsfw = not opt.horde_sfw if opt.horde_sfw else cd.horde_nsfw
            except AttributeError:
                horde_nsfw = True
            try:
                horde_censor_nsfw = opt.horde_censor_nsfw if opt.horde_censor_nsfw else cd.horde_censor_nsfw
            except AttributeError:
                horde_censor_nsfw = False
            try:
                horde_blacklist = opt.horde_blacklist if opt.horde_blacklist else cd.horde_blacklist
            except AttributeError:
                horde_blacklist = []
            try:
                horde_censorlist = opt.horde_censorlist if opt.horde_censorlist else cd.horde_censorlist
            except AttributeError:
                horde_censorlist = []
            if horde_max_power < 2:
                horde_max_power = 2
            horde_max_pixels = 64*64*8*horde_max_power
            logger.info(f"Joining Horde with parameters: Server Name '{horde_name}'. Horde URL '{horde_url}'. Max Pixels {horde_max_pixels}")

            try:
                thread = threading.Thread(target=run_bridge(1, horde_api_key, horde_name, horde_url,
                                                            horde_priority_usernames, horde_max_pixels,
                                                            horde_nsfw, horde_censor_nsfw, horde_blacklist,
                                                            horde_censorlist), args=())
                thread.daemon = True
                thread.start()
                #run_bridge(1, horde_api_key, horde_name, horde_url, horde_priority_usernames, horde_max_pixels, horde_nsfw, horde_censor_nsfw, horde_blacklist, horde_censorlist)
            except KeyboardInterrupt:
                print(f"Keyboard Interrupt Received. Ending Bridge")
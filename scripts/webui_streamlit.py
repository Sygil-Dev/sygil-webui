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
#import streamlit as st

# We import hydralit like this to replace the previous stuff
# we had with native streamlit as it lets ur replace things 1:1
from sd_utils import st, hc, load_configs, load_css, set_logger_verbosity,\
     logger, quiesce_logger, set_page_title, threading, random

# streamlit imports
import streamlit_nested_layout

#streamlit components section
#from st_on_hover_tabs import on_hover_tabs
from streamlit_server_state import server_state, server_state_lock

#other imports
import argparse
from sd_utils.bridge import run_bridge

# import custom components
from custom_components import draggable_number_input

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

@logger.catch(reraise=True)
def layout():
        """Layout functions to define all the streamlit layout here."""
        if not st.session_state["defaults"].debug.enable_hydralit:
            st.set_page_config(page_title="Stable Diffusion Playground", layout="wide", initial_sidebar_state="collapsed")

        #app = st.HydraApp(title='Stable Diffusion WebUI', favicon="", sidebar_state="expanded", layout="wide",
                                        #hide_streamlit_markers=False, allow_url_nav=True , clear_cross_app_sessions=False)


        # load css as an external file, function has an option to local or remote url. Potential use when running from cloud infra that might not have access to local path.
        load_css(True, 'frontend/css/streamlit.main.css')

        #
        # specify the primary menu definition
        menu_data = [
            {'id': 'Stable Diffusion', 'label': 'Stable Diffusion', 'icon': 'bi bi-grid-1x2-fill'},
            {'id': 'Train','label':"Train", 'icon': "bi bi-lightbulb-fill", 'submenu':[
                {'id': 'Textual Inversion', 'label': 'Textual Inversion', 'icon': 'bi bi-lightbulb-fill'},
                {'id': 'Fine Tunning', 'label': 'Fine Tunning', 'icon': 'bi bi-lightbulb-fill'},
                ]},
            {'id': 'Model Manager', 'label': 'Model Manager', 'icon': 'bi bi-cloud-arrow-down-fill'},
            {'id': 'Tools','label':"Tools", 'icon': "bi bi-tools", 'submenu':[
                {'id': 'API Server', 'label': 'API Server', 'icon': 'bi bi-server'},
                {'id': 'Barfi/BaklavaJS', 'label': 'Barfi/BaklavaJS', 'icon': 'bi bi-diagram-3-fill'},
                #{'id': 'API Server', 'label': 'API Server', 'icon': 'bi bi-server'},
                ]},
            {'id': 'Settings', 'label': 'Settings', 'icon': 'bi bi-gear-fill'},
        ]

        over_theme = {'txc_inactive': '#FFFFFF', "menu_background":'#000000'}

        menu_id = hc.nav_bar(
            menu_definition=menu_data,
            #home_name='Home',
            #login_name='Logout',
            hide_streamlit_markers=False,
            override_theme=over_theme,
            sticky_nav=True,
            sticky_mode='pinned',
        )

        #
        #if menu_id == "Home":
            #st.info("Under Construction. :construction_worker:")

        if menu_id == "Stable Diffusion":
            # set the page url and title
            #st.experimental_set_query_params(page='stable-diffusion')
            try:
                set_page_title("Stable Diffusion Playground")
            except NameError:
                st.experimental_rerun()

            txt2img_tab, img2img_tab, txt2vid_tab, img2txt_tab, post_processing_tab, concept_library_tab = st.tabs(["Text-to-Image", "Image-to-Image",
                                                                                                                    #"Inpainting",
                                                                                                                    "Text-to-Video", "Image-To-Text",
                                                                                                                    "Post-Processing","Concept Library"])
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

            with post_processing_tab:
                from post_processing import layout
                layout()

            with concept_library_tab:
                from sd_concept_library import layout
                layout()

        #
        elif menu_id == 'Model Manager':
            set_page_title("Model Manager - Stable Diffusion Playground")

            from ModelManager import layout
            layout()

        elif menu_id == 'Textual Inversion':
            from textual_inversion import layout
            layout()

        elif menu_id == 'Fine Tunning':
            #from textual_inversion import layout
            #layout()
            st.info("Under Construction. :construction_worker:")

        elif menu_id == 'API Server':
            set_page_title("API Server - Stable Diffusion Playground")
            from APIServer import layout
            layout()

        elif menu_id == 'Barfi/BaklavaJS':
            set_page_title("Barfi/BaklavaJS - Stable Diffusion Playground")
            from barfi_baklavajs import layout
            layout()

        elif menu_id == 'Settings':
            set_page_title("Settings - Stable Diffusion Playground")

            from Settings import layout
            layout()

        # calling dragable input component module at the end, so it works on all pages
        draggable_number_input.load()


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
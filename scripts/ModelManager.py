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
from sd_utils import *

# streamlit imports


#other imports

# Temp imports 


# end of imports
#---------------------------------------------------------------------------------------------------------------

def layout():
    #search = st.text_input(label="Search", placeholder="Type the name of the model you want to search for.", help="")
    
    colms = st.columns((1, 3, 5, 5))
    columns = ["№",'Model Name','Save Location','Download Link']
    
    models = st.session_state["defaults"].model_manager.models

    for col, field_name in zip(colms, columns):
        # table header
        col.write(field_name)
        
    for x, model_name in enumerate(models):
        col1, col2, col3, col4 = st.columns((1, 3, 4, 6))
        col1.write(x)  # index
        col2.write(models[model_name]['model_name'])
        col3.write(models[model_name]['save_location'])
        col4.write(models[model_name]['download_link'])    
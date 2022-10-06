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
def download_file(file_name, file_path, file_url):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    if not os.path.exists(os.path.join(file_path , file_name)):
        print('Downloading ' + file_name + '...')
        # TODO - add progress bar in streamlit
        # download file with `requests``
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(os.path.join(file_path, file_name), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    else:
        print(file_name + ' already exists.')

def download_model(models, model_name):
    """ Download all files from model_list[model_name] """
    for file in models[model_name]:
        download_file(file['file_name'], file['file_path'], file['file_url'])
    return


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
        with col4:
            files_exist = 0
            for file in models[model_name]['files']:
                if "save_location" in models[model_name]['files'][file]:
                    os.path.exists(os.path.join(models[model_name]['files'][file]['save_location'] , models[model_name]['files'][file]['file_name']))
                    files_exist += 1
                elif os.path.exists(os.path.join(models[model_name]['save_location'] , models[model_name]['files'][file]['file_name'])):
                    files_exist += 1
            files_needed = []
            for file in models[model_name]['files']:
                if "save_location" in models[model_name]['files'][file]:
                    if not os.path.exists(os.path.join(models[model_name]['files'][file]['save_location'] , models[model_name]['files'][file]['file_name'])):
                        files_needed.append(file)
                elif not os.path.exists(os.path.join(models[model_name]['save_location'] , models[model_name]['files'][file]['file_name'])):
                    files_needed.append(file)
            if len(files_needed) > 0:
                if st.button('Download', key=models[model_name]['model_name'], help='Download ' + models[model_name]['model_name']):
                    for file in files_needed:
                        if "save_location" in models[model_name]['files'][file]:
                            download_file(models[model_name]['files'][file]['file_name'], models[model_name]['files'][file]['save_location'], models[model_name]['files'][file]['download_link'])
                        else:
                            download_file(models[model_name]['files'][file]['file_name'], models[model_name]['save_location'], models[model_name]['files'][file]['download_link'])
                else:
                    st.empty()
            else:
                st.write('✅')
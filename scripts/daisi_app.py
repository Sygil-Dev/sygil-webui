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
import os, subprocess
import yaml


try:
	with open("environment.yaml") as file_handle:
		environment_data = yaml.load(file_handle)
except FileNotFoundError:
	with open("../environment.yaml") as file_handle:
		environment_data = yaml.load(file_handle)	

for dependency in environment_data["dependencies"]:
	package_name, package_version = dependency.split("=")
	os.system("pip install {}=={}".format(package_name, package_version))

try:
	subprocess.run(['python', '-m', 'streamlit', "run" ,"../scripts/webui_streamlit.py", "--theme.base dark"], stdout=subprocess.DEVNULL) 
except FileExistsError:
	subprocess.run(['python', '-m', 'streamlit', "run" ,"scripts/webui_streamlit.py", "--theme.base dark"], stdout=subprocess.DEVNULL) 
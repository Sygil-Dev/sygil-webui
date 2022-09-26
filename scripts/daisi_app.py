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
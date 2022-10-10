import os, subprocess
import yaml

print (os.getcwd)

try:
	with open("environment.yaml") as file_handle:
		environment_data = yaml.safe_load(file_handle, Loader=yaml.FullLoader)
except FileNotFoundError:
	try:
		with open(os.path.join("..", "environment.yaml")) as file_handle:
			environment_data = yaml.safe_load(file_handle, Loader=yaml.FullLoader)
	except:
		pass

try:
	for dependency in environment_data["dependencies"]:
		package_name, package_version = dependency.split("=")
		os.system("pip install {}=={}".format(package_name, package_version))
except:
	pass

try:
	subprocess.run(['python', '-m', 'streamlit', "run" ,os.path.join("..","scripts/webui_streamlit.py"), "--theme.base dark"], stdout=subprocess.DEVNULL)
except FileExistsError:
	subprocess.run(['python', '-m', 'streamlit', "run" ,"scripts/webui_streamlit.py", "--theme.base dark"], stdout=subprocess.DEVNULL)
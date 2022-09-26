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

import os, subprocess, shutil
from huggingface_hub import HfApi
from git import Repo, RemoteProgress

class CloneProgress(RemoteProgress):
	def update(self, op_code, cur_count, max_count=None, message=''):
		if message:
			print(message)
api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)
models = []

print ("Downloading the sd concept library from the huggingface site.")
for model in models_list:
	model_content = {}
	model_id = model.modelId
	url = f"https://huggingface.co/{model_id}"
	try:
		if not os.path.exists(os.path.join("../models/custom", model_id)):
			subprocess.run(['git', 'lfs', 'install'], stdout=subprocess.DEVNULL)
			Repo.clone_from(url, os.path.join("../models/custom", model_id), progress=CloneProgress())	
		#else:
			#repo = Repo(os.path.join("../models/custom", model_id))
			#repo.git.stash('save')
			#repo.git.pull()
			
		subprocess.run(['git', 'lfs', 'uninstall'], stdout=subprocess.DEVNULL) # uninstall LFS
		os.remove(os.path.join("../models/custom", model_id, '.gitattributes')) if os.path.exists(os.path.join("../models/custom", model_id, '.gitattributes')) else None # remove the .gitattributes so files don't use LFS
		subprocess.run(['rm', '-rf', os.path.join("../models/custom", model_id,'.git')]) if os.path.exists(os.path.join("../models/custom", model_id, '.git')) else None   # remove all the .git folders as we dont need them.
		# get the folder size and delete it if its larger than 100mb
		size = 0		
		for ele in os.scandir(os.path.join("../models/custom", model_id)): # get folder size
			size+=os.stat(ele).st_size		
		if size > 100000000: # if the folder is larger than 100mb delete it.
			shutil.rmtree(os.path.join("../models/custom", model_id), ignore_errors=True)
	except:
		pass

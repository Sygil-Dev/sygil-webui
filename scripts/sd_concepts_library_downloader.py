import os
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
		Repo.clone_from(url, os.path.join("../models/custom", model_id), progress=CloneProgress())	
	except:
		pass


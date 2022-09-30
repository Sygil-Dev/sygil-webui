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

"""
CLIP Interrogator made by @pharmapsychotic modified to work with our WebUI.

# CLIP Interrogator by @pharmapsychotic 
Twitter: https://twitter.com/pharmapsychotic
Github: https://github.com/pharmapsychotic/clip-interrogator

Description:
What do the different OpenAI CLIP models see in an image? What might be a good text prompt to create similar images using CLIP guided diffusion
or another text to image model? The CLIP Interrogator is here to get you answers!

Please consider buying him a coffee via [ko-fi](https://ko-fi.com/pharmapsychotic) or following him on [twitter](https://twitter.com/pharmapsychotic).

And if you're looking for more Ai art tools check out my [Ai generative art tools list](https://pharmapsychotic.com/tools.html).

"""

#
# base webui import and utils.
from sd_utils import *

# streamlit imports
import streamlit_nested_layout

#streamlit components section

#other imports
import clip
import gc
import os
import pandas as pd
#import requests
import torch
from PIL import Image
#from torch import nn
#from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from ldm.models.blip import blip_decoder

# end of imports
#---------------------------------------------------------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

blip_image_eval_size = 256
#blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'   

def load_blip_model():
	blip_model = blip_decoder(pretrained="models/blip/model__base_caption.pth", image_size=blip_image_eval_size, vit='base', med_config="configs/blip/med_config.json")
	blip_model.eval()

	return blip_model

def load_clip_model(clip_model_name):
	import clip

	model, preprocess = clip.load(clip_model_name)
	model.eval()
	model = model.to(device)

	return model, preprocess

def generate_caption(pil_image):
	blip_model = blip_decoder(pretrained="models/blip/model__base_caption.pth", image_size=blip_image_eval_size, vit='base', med_config="configs/blip/med_config.json")
	blip_model.eval()
	blip_model = blip_model.to(device)
	
	gpu_image = transforms.Compose([
	    transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
	    transforms.ToTensor(),
	    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
	    ])(pil_image).unsqueeze(0).to(device)

	with torch.no_grad():
		caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
	return caption[0]

def load_list(filename):
	with open(filename, 'r', encoding='utf-8', errors='replace') as f:
		items = [line.strip() for line in f.readlines()]
	return items

def rank(model, image_features, text_array, top_count=1):
	top_count = min(top_count, len(text_array))
	text_tokens = clip.tokenize([text for text in text_array]).cuda()
	with torch.no_grad():
		text_features = model.encode_text(text_tokens).float()
	text_features /= text_features.norm(dim=-1, keepdim=True)

	similarity = torch.zeros((1, len(text_array))).to(device)
	for i in range(image_features.shape[0]):
		similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
	similarity /= image_features.shape[0]

	top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)  
	return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

def interrogate(image, models):
	print ("Generating Caption")
	st.session_state["log_message"].code("Generating Caption", language='')
	caption = generate_caption(image)
	
	if len(models) == 0:
		print(f"\n\n{caption}")
		return

	table = []
	bests = [[('',0)]]*5
	for model_name in models:
		st.session_state["log_message"].code(f"Interrogating with {model_name}...", language='')
		model, preprocess = load_clip_model(model_name)
		model.cuda().eval()

		images = preprocess(image).unsqueeze(0).cuda()
		with torch.no_grad():
			image_features = model.encode_image(images).float()
		image_features /= image_features.norm(dim=-1, keepdim=True)

		ranks = [
		    rank(model, image_features, server_state["mediums"]),
		    rank(model, image_features, ["by "+artist for artist in server_state["artists"]]),
		    rank(model, image_features, server_state["trending_list"]),
		    rank(model, image_features, server_state["movements"]),
		    rank(model, image_features, server_state["flavors"], top_count=3)
		]

		for i in range(len(ranks)):
			confidence_sum = 0
			for ci in range(len(ranks[i])):
				confidence_sum += ranks[i][ci][1]
			if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
				bests[i] = ranks[i]

		row = [model_name]
		for r in ranks:
			row.append(', '.join([f"{x[0]} ({x[1]:0.1f}%)" for x in r]))

		table.append(row)

		del model
		gc.collect()
		
	st.session_state["prediction_table"].dataframe(pd.DataFrame(table, columns=["Model", "Medium", "Artist", "Trending", "Movement", "Flavors"]))

	flaves = ', '.join([f"{x[0]}" for x in bests[4]])
	medium = bests[0][0][0]
	if caption.startswith(medium):
		st.session_state["text_result"].code(f"\n\n{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}", language="")
	else:
		st.session_state["text_result"].code(f"\n\n{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}", language="")
	
	st.session_state["log_message"].code("Finished Interrogating.", language="")
#

def img2txt():
	data_path = "data/"
	
	server_state["artists"] = load_list(os.path.join(data_path, 'img2txt', 'artists.txt'))
	server_state["flavors"] = load_list(os.path.join(data_path, 'img2txt', 'flavors.txt'))
	server_state["mediums"] = load_list(os.path.join(data_path, 'img2txt', 'mediums.txt'))
	server_state["movements"] = load_list(os.path.join(data_path, 'img2txt', 'movements.txt'))
	server_state["sites"] = load_list(os.path.join(data_path, 'img2txt', 'sites.txt'))
	
	server_state["trending_list"] = [site for site in server_state["sites"]]
	server_state["trending_list"].extend(["trending on "+site for site in server_state["sites"]])
	server_state["trending_list"].extend(["featured on "+site for site in server_state["sites"]])
	server_state["trending_list"].extend([site+" contest winner" for site in server_state["sites"]])
	
	#image_path_or_url = "https://i.redd.it/e2e8gimigjq91.jpg"
	
	models = []
	
	if st.session_state["ViTB32"]:
		models.append('ViT-B/32')
	if st.session_state['ViTB16']:
		models.append('ViT-B/16')
	if st.session_state["ViTL14"]: 
		models.append('ViT-L/14')
	if st.session_state["ViTL14_336px"]:
		models.append('ViT-L/14@336px')
	if st.session_state["RN101"]:
		models.append('RN101')
	if st.session_state["RN50"]:
		models.append('RN50')
	if st.session_state["RN50x4"]:
		models.append('RN50x4')
	if st.session_state["RN50x16"]:
		models.append('RN50x16')
	if st.session_state["RN50x64"]:
		models.append('RN50x64')
	
	#if str(image_path_or_url).startswith('http://') or str(image_path_or_url).startswith('https://'):
		#image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
	#else:
		#image = Image.open(image_path_or_url).convert('RGB')
	
	#thumb = st.session_state["uploaded_image"].image.copy()
	#thumb.thumbnail([blip_image_eval_size, blip_image_eval_size])
	#display(thumb)
	
	interrogate(st.session_state["uploaded_image"].pil_image, models=models)

#
def layout():
	#set_page_title("Image-to-Text - Stable Diffusion WebUI")
	#st.info("Under Construction. :construction_worker:")	
	
	with st.form("img2txt-inputs"):
		st.session_state["generation_mode"] = "img2txt"

		#st.write("---")
		# creating the page layout using columns
		col1, col2, col3 = st.columns([1,2,1], gap="large")   	
		
		with col1:
			#url = st.text_area("Input Text","")
			#url = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")
			#st.subheader("Input Image")
			st.session_state["uploaded_image"] = st.file_uploader('Input Image', type=['png', 'jpg', 'jpeg'])
					
			st.subheader("CLIP models")	
			with st.expander("Stable Diffusion", expanded=True):
				st.session_state["ViTL14"] = st.checkbox("ViTL14", value=True, help="For StableDiffusion you can just use ViTL14.")
			
			with st.expander("Others"):
				st.info("For DiscoDiffusion and JAX enable all the same models here as you intend to use when generating your images.")
				st.session_state["ViTL14_336px"] = st.checkbox("ViTL14_336px", value=False, help="ViTL14_336px model.")			
				st.session_state["ViTB16"] = st.checkbox("ViTB16", value=False, help="ViTB16 model.")			
				st.session_state["ViTB32"] = st.checkbox("ViTB32", value=False, help="ViTB32 model.")
				st.session_state["RN50"] = st.checkbox("RN50", value=False, help="RN50 model.")			
				st.session_state["RN50x4"] = st.checkbox("RN50x4", value=False, help="RN50x4 model.")			
				st.session_state["RN50x16"] = st.checkbox("RN50x16", value=False, help="RN50x16 model.")			
				st.session_state["RN50x64"] = st.checkbox("RN50x64", value=False, help="RN50x64 model.")
				st.session_state["RN101"] = st.checkbox("RN101", value=False, help="RN101 model.")			
				
			
			with col2:
				st.subheader("Image")
				
				st.form_submit_button("Refresh",
									  help='Refresh the image preview to show your uploaded image instead of the default placeholder.')
				st.session_state["input_image_preview"] = st.empty()
				
				if st.session_state["uploaded_image"]:				
					st.session_state["uploaded_image"].pil_image = Image.open(st.session_state["uploaded_image"])#.convert('RGBA')
					#new_img = image.resize((width, height))
					st.session_state["input_image_preview"].image(st.session_state["uploaded_image"].pil_image, clamp=True)	
				else:
					#st.session_state["input_image_preview"].code('', language="")
					st.image("images/streamlit/img2txt_placeholder.png", clamp=True)
				
			with col3:
				st.subheader("Logs:")
				
				st.session_state["log_message"] = st.empty()
				st.session_state["log_message"].code('', language="")
		
		#
		# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
		#generate_col1.title("")
		#generate_col1.title("")
		generate_button = st.form_submit_button("Generate!")		
				
		#
		st.write("---")
		
		with st.container():
			st.subheader("Image To Text Result")		
			
			st.session_state["prediction_table"] = st.empty()
			st.session_state["prediction_table"].table()
			
			st.session_state["text_result"] = st.empty()
			st.session_state["text_result"].code('', language="")

	
	if generate_button:		
		# run clip interrogator
		img2txt()
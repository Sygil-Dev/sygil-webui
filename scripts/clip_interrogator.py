
#@title Setup
#!pip3 install ftfy regex tqdm transformers==4.15.0 timm==0.4.12 fairscale==0.4.4
#!pip3 install git+https://github.com/openai/CLIP.git
#!git clone https://github.com/pharmapsychotic/clip-interrogator.git
#!git clone https://github.com/salesforce/BLIP
#%cd /content/BLIP

import clip
import gc
#import numpy as np
import os
import pandas as pd
import requests
import torch
#import torchvision.transforms as T
#import torchvision.transforms.functional as TF

from IPython.display import display
from PIL import Image
#from torch import nn
#from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from ldm.models.blip import blip_decoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

blip_image_eval_size = 384
blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'        
blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit='base')
blip_model.eval()
blip_model = blip_model.to(device)

def generate_caption(pil_image):
	gpu_image = transforms.Compose([
	    transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
	    transforms.ToTensor(),
	    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
	    ])(image).unsqueeze(0).to(device)

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
	caption = generate_caption(image)
	if len(models) == 0:
		print(f"\n\n{caption}")
		return

	table = []
	bests = [[('',0)]]*5
	for model_name in models:
		print(f"Interrogating with {model_name}...")
		model, preprocess = clip.load(model_name)
		model.cuda().eval()

		images = preprocess(image).unsqueeze(0).cuda()
		with torch.no_grad():
			image_features = model.encode_image(images).float()
		image_features /= image_features.norm(dim=-1, keepdim=True)

		ranks = [
		    rank(model, image_features, mediums),
		    rank(model, image_features, ["by "+artist for artist in artists]),
		    rank(model, image_features, trending_list),
		    rank(model, image_features, movements),
		    rank(model, image_features, flavors, top_count=3)
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
	display(pd.DataFrame(table, columns=["Model", "Medium", "Artist", "Trending", "Movement", "Flavors"]))

	flaves = ', '.join([f"{x[0]}" for x in bests[4]])
	medium = bests[0][0][0]
	if caption.startswith(medium):
		print(f"\n\n{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}")
	else:
		print(f"\n\n{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}")

data_path = "../clip-interrogator/data/"

artists = load_list(os.path.join(data_path, 'artists.txt'))
flavors = load_list(os.path.join(data_path, 'flavors.txt'))
mediums = load_list(os.path.join(data_path, 'mediums.txt'))
movements = load_list(os.path.join(data_path, 'movements.txt'))

sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
trending_list = [site for site in sites]
trending_list.extend(["trending on "+site for site in sites])
trending_list.extend(["featured on "+site for site in sites])
trending_list.extend([site+" contest winner" for site in sites])

#@title Interrogate!

#@markdown 

#@markdown #####**Image:**

image_path_or_url = "https://i.redd.it/e2e8gimigjq91.jpg" #@param {type:"string"}

#@markdown 

#@markdown #####**CLIP models:**

#@markdown For [StableDiffusion](https://stability.ai/blog/stable-diffusion-announcement) you can just use ViTL14<br>
#@markdown For [DiscoDiffusion](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb) and 
#@markdown [JAX](https://colab.research.google.com/github/huemin-art/jax-guided-diffusion/blob/v2.7/Huemin_Jax_Diffusion_2_7.ipynb) enable all the same models here as you intend to use when generating your images

ViTB32 = True #@param{type:"boolean"}
ViTB16 = True #@param{type:"boolean"}
ViTL14 = False #@param{type:"boolean"}
ViTL14_336px = False #@param{type:"boolean"}
RN101 = False #@param{type:"boolean"}
RN50 = True #@param{type:"boolean"}
RN50x4 = False #@param{type:"boolean"}
RN50x16 = False #@param{type:"boolean"}
RN50x64 = False #@param{type:"boolean"}

models = []
if ViTB32: models.append('ViT-B/32')
if ViTB16: models.append('ViT-B/16')
if ViTL14: models.append('ViT-L/14')
if ViTL14_336px: models.append('ViT-L/14@336px')
if RN101: models.append('RN101')
if RN50: models.append('RN50')
if RN50x4: models.append('RN50x4')
if RN50x16: models.append('RN50x16')
if RN50x64: models.append('RN50x64')

if str(image_path_or_url).startswith('http://') or str(image_path_or_url).startswith('https://'):
	image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
else:
	image = Image.open(image_path_or_url).convert('RGB')

thumb = image.copy()
thumb.thumbnail([blip_image_eval_size, blip_image_eval_size])
display(thumb)

interrogate(image, models=models)

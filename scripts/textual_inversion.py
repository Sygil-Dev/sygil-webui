# base webui import and utils.
from webui_streamlit import st
from sd_utils import *

# streamlit imports


#other imports
#from transformers import CLIPTextModel, CLIPTokenizer

# Temp imports


# end of imports
#---------------------------------------------------------------------------------------------------------------

#def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
	
	#loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

	## separate token and the embeds
	#print (loaded_learned_embeds)
	#trained_token = list(loaded_learned_embeds.keys())[0]
	#embeds = loaded_learned_embeds[trained_token]

	## cast to dtype of text_encoder
	#dtype = text_encoder.get_input_embeddings().weight.dtype
	#embeds.to(dtype)

	## add the token in tokenizer
	#token = token if token is not None else trained_token
	#num_added_tokens = tokenizer.add_tokens(token)
	#i = 1
	#while(num_added_tokens == 0):
		#print(f"The tokenizer already contains the token {token}.")
		#token = f"{token[:-1]}-{i}>"
		#print(f"Attempting to add the token {token}.")
		#num_added_tokens = tokenizer.add_tokens(token)
		#i+=1

	## resize the token embeddings
	#text_encoder.resize_token_embeddings(len(tokenizer))

	## get the id for the token and assign the embeds
	#token_id = tokenizer.convert_tokens_to_ids(token)
	#text_encoder.get_input_embeddings().weight.data[token_id] = embeds
	#return token

##def token_loader()
#learned_token = load_learned_embed_in_clip(f"models/custom/embeddings/Custom Ami.pt", st.session_state.pipe.text_encoder, st.session_state.pipe.tokenizer, "*")
#model_content["token"] = learned_token
#models.append(model_content)

model_id = "./models/custom/embeddings/"

def layout():
	st.write("Textual Inversion")
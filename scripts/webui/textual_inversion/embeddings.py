from util.imports import *

def load_embeddings(fp):
	if fp is not None and hasattr(st.session_state["model"], "embedding_manager"):
		st.session_state["model"].embedding_manager.load(fp['name'])

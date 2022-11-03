import os
import streamlit.components.v1 as components

def load(pixel_per_step = 50):
	parent_dir = os.path.dirname(os.path.abspath(__file__))
	file = os.path.join(parent_dir, "main.js")

	with open(file) as f:
		javascript_main = f.read()
		javascript_main = javascript_main.replace("%%pixelPerStep%%",str(pixel_per_step))
		components.html(f"<script>{javascript_main}</script>")
import os
from collections import defaultdict
import streamlit.components.v1 as components

# where to save the downloaded key_phrases
key_phrases_file = "data/tags/key_phrases.json"
# the loaded key phrase json as text
key_phrases_json = ""
# where to save the downloaded key_phrases
thumbnails_file = "data/tags/thumbnails.json"
# the loaded key phrase json as text
thumbnails_json = ""

def init():
	global key_phrases_json, thumbnails_json
	with open(key_phrases_file) as f:
		key_phrases_json = f.read()
	with open(thumbnails_file) as f:
		thumbnails_json = f.read()

def suggestion_area(placeholder):
	# get component path
	parent_dir = os.path.dirname(os.path.abspath(__file__))
	# get file paths
	javascript_file = os.path.join(parent_dir, "main.js")
	stylesheet_file = os.path.join(parent_dir, "main.css")
	parent_stylesheet_file = os.path.join(parent_dir, "parent.css")

	# load file texts
	with open(javascript_file) as f:
		javascript_main = f.read()
	with open(stylesheet_file) as f:
		stylesheet_main = f.read()
	with open(parent_stylesheet_file) as f:
		parent_stylesheet = f.read()

	# add suggestion area div box
	html = "<div id='scroll_area' class='st-bg'><div id='suggestion_area'>javascript failed</div></div>"
	# add loaded style
	html += f"<style>{stylesheet_main}</style>"
	# set default variables
	html += f"<script>var thumbnails = {thumbnails_json};\nvar keyPhrases = {key_phrases_json};\nvar parentCSS = `{parent_stylesheet}`;\nvar placeholder='{placeholder}';</script>"
	# add main java script
	html += f"\n<script>{javascript_main}</script>"
	# add component to site
	components.html(html, width=None, height=None, scrolling=True)
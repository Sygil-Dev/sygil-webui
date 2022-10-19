import os, requests, io, csv, json
from collections import defaultdict
import streamlit.components.v1 as components

# the google sheet document id (its in the url)
doc_id = "1WoMEpGiZia0AfngkT6sRUhGOwczBMyoslk2jhYNUoiw"
# the page to get from the document (also visible in the url, the gid)
key_phrase_sheet_id = "723922433"
# where to save the downloaded key_phrases
key_phrases_file = "data/tags/key_phrases.json"
# the loaded key phrase json as text
key_phrases_json = ""

def download_and_save_as_json(url):
	global key_phrases_json

	# download
	r = requests.get(url)

	# we need the bytes to decode utf-8 and use it in a stringIO
	csv_bytes = r.content
	# stringIO for parsing via csv.DictReader
	str_file = io.StringIO(csv_bytes.decode('utf-8'), newline='\n')

	reader = csv.DictReader(str_file, delimiter=',', quotechar='"')

	# structure data in usable format (columns as arrays)
	columnwise_table = defaultdict(list)
	for row in reader:
		for col, dat in row.items():
			stripped = dat.strip()
			if stripped != "":
				columnwise_table[col].append(dat.strip())

	# dump the data as json
	key_phrases_json = json.dumps(columnwise_table, indent=4)

	# save json so we don't need to download it every time
	with open(key_phrases_file, 'w', encoding='utf-8') as jsonf:
		jsonf.write(key_phrases_json)

def update_key_phrases():
	url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={key_phrase_sheet_id}"
	download_and_save_as_json(url)

def init():
	global key_phrases_json
	if not os.path.isfile(key_phrases_file):
		update_key_phrases()
	else:
		with open(key_phrases_file) as f:
			key_phrases_json = f.read()

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
	html = "<div id='suggestion_area'>javascript failed</div>"
	# add loaded style
	html += f"<style>{stylesheet_main}</style>"
	# set default variables
	html += f"<script>var keyPhrases = {key_phrases_json};\nvar parentCSS = `{parent_stylesheet}`;\nvar placeholder='{placeholder}';</script>"
	# add main java script
	html += f"\n<script>{javascript_main}</script>"
	# add component to site
	components.html(html, width=None, height=None, scrolling=True)
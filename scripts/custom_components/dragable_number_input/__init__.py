import os
import streamlit.components.v1 as components

def load():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(parent_dir, "main.js")

    with open(file) as f:
        javascript_main = f.read()
        components.html(f"<script>{javascript_main}</script>")
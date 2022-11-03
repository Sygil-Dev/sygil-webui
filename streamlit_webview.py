import os, webview
from streamlit.web import bootstrap
from streamlit import config as _config

webview.create_window('Sygil', 'http://localhost:8501', width=1000, height=800, min_size=(500, 500))
webview.start()

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'scripts/webui_streamlit.py')

_config.set_option("server.headless", True)
args = []

#streamlit.cli.main_run(filename, args)
bootstrap.run(filename,'',args, flag_options={})
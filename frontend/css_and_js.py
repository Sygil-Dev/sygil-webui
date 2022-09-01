from os import path
import json


def readTextFile(*args):
    dir = path.dirname(__file__)
    entry = path.join(dir, *args)
    with open(entry, "r", encoding="utf8") as f:
        data = f.read()
    return data


def css(opt):
    styling = readTextFile("css", "styles.css")
    if not opt.no_progressbar_hiding:
        styling += readTextFile("css", "no_progress_bar.css")
    return styling


def js(opt):
    data = readTextFile("js", "index.js")
    data = "(z) => {" + data + "; return z ?? [] }"
    return data


# Wrap the typical SD method call into async closure for ease of use
# Supplies the js function with a params object
# That includes all the passed arguments and input from Gradio: x
# Example call in Gradio component's event handler (pass the result to _js arg):
# _js=call_JS("myJsMethod", arg1="string", arg2=100, arg3=[])
def call_JS(sd_method, **kwargs):
    param_str = json.dumps(kwargs)
    return f"async (x) => {{ return await SD.{sd_method}({{ x, ...{param_str} }}) ?? []; }}"

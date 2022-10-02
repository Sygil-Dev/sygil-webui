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
# ATTENTION: x is an array of values of all components passed to your
# python event handler
# Example call in Gradio component's event handler (pass the result to _js arg):
# _js=call_JS("myJsMethod", arg1="string", arg2=100, arg3=[])
def call_JS(sd_method, **kwargs):
    param_str = json.dumps(kwargs)
    return f"async (...x) => {{ return await SD.{sd_method}({{ x, ...{param_str} }}) ?? []; }}"

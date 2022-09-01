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
    # TODO: @altryne restore this before merge
    if not opt.no_progressbar_hiding:
         styling += readTextFile("css", "no_progress_bar.css")
    return styling


def js(opt):
    data = readTextFile("js", "index.js")
    data = "(z) => {" + data + "; return z ?? [] }"
    return data



def js_painterro_launch(to_id):
    return w(f"Painterro.init('{to_id}')")

def js_move_image(from_id, to_id):
    return w(f"moveImageFromGallery('{from_id}', '{to_id}')")

def js_copy_to_clipboard(from_id):
    return w(f"copyImageFromGalleryToClipboard('{from_id}')")

def js_img2img_submit(prompt_row_id):
    return w(f"clickFirstVisibleButton('{prompt_row_id}')")

# TODO : @altryne fix this to the new JS format
js_copy_txt2img_output = "(x) => {navigator.clipboard.writeText(document.querySelector('gradio-app').shadowRoot.querySelector('#highlight .textfield').textContent.replace(/\s+/g,' ').replace(/: /g,':'))}"



js_parse_prompt ="""
(txt2img_prompt, txt2img_width, txt2img_height, txt2img_steps, txt2img_seed, txt2img_batch_count, txt2img_cfg) => {
    
const prompt_input = document.querySelector('gradio-app').shadowRoot.querySelector('#prompt_input [data-testid="textbox"]');
const multiline = document.querySelector('gradio-app').shadowRoot.querySelector('#submit_on_enter label:nth-child(2)')
if (prompt_input.scrollWidth > prompt_input.clientWidth + 10 ) {
   multiline.click(); 
}


let height_match =  /(?:-h|-H|--height|height)[ :]?(?<height>\d+) /.exec(txt2img_prompt);
if (height_match) {
    txt2img_height = Math.round(height_match.groups.height / 64) * 64;
    txt2img_prompt = txt2img_prompt.replace(height_match[0], '');
}
let width_match =  /(?:-w|-W|--width|width)[ :]?(?<width>\d+) /.exec(txt2img_prompt);
if (width_match) {
    txt2img_width = Math.round(width_match.groups.width / 64) * 64;
    txt2img_prompt = txt2img_prompt.replace(width_match[0], '');
}
let steps_match =  /(?:-s|--steps|steps)[ :]?(?<steps>\d+) /.exec(txt2img_prompt);
if (steps_match) {
    txt2img_steps = steps_match.groups.steps.trim();
    txt2img_prompt = txt2img_prompt.replace(steps_match[0], '');
}
let seed_match =  /(?:-S|--seed|seed)[ :]?(?<seed>\d+) /.exec(txt2img_prompt);
if (seed_match) {
    txt2img_seed = seed_match.groups.seed;
    txt2img_prompt = txt2img_prompt.replace(seed_match[0], '');
}
let batch_count_match =  /(?:-n|-N|--number|number)[ :]?(?<batch_count>\d+) /.exec(txt2img_prompt);
if (batch_count_match) {
    txt2img_batch_count = batch_count_match.groups.batch_count;
    txt2img_prompt = txt2img_prompt.replace(batch_count_match[0], '');
}
let cfg_scale_match =  /(?:-c|-C|--cfg-scale|cfg_scale|cfg)[ :]?(?<cfgscale>\d\.?\d+?) /.exec(txt2img_prompt);
if (cfg_scale_match) {
    txt2img_cfg = parseFloat(cfg_scale_match.groups.cfgscale).toFixed(1);
    txt2img_prompt = txt2img_prompt.replace(cfg_scale_match[0], '');
}
let sampler_match =  /(?:-A|--sampler|sampler)[ :]?(?<sampler>\w+) /.exec(txt2img_prompt);
if (sampler_match) {
    
    txt2img_prompt = txt2img_prompt.replace(sampler_match[0], '');
}

return [txt2img_prompt, parseInt(txt2img_width), parseInt(txt2img_height), parseInt(txt2img_steps), txt2img_seed, parseInt(txt2img_batch_count), parseFloat(txt2img_cfg)];
}
"""


# @altryne this came up as conflict, still needed or no?
# Wrap the typical SD method call into async closure for ease of use
# Supplies the js function with a params object
# That includes all the passed arguments and input from Gradio: x
# Example call in Gradio component's event handler (pass the result to _js arg):
# _js=call_JS("myJsMethod", arg1="string", arg2=100, arg3=[])
def call_JS(sd_method, **kwargs):
    param_str = json.dumps(kwargs)
    return f"async (x) => {{ return await SD.{sd_method}({{ x, ...{param_str} }}) ?? []; }}"

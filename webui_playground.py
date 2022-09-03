import gradio as gr
import time
import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re

from frontend.frontend import draw_gradio_ui
from frontend.ui_functions import resize_image

"""
This file is here to play around with the interface without loading the whole model 

TBD - extract all the UI into this file and import from the main webui. 
"""

GFPGAN = True
RealESRGAN = True


def run_goBIG():
    pass


def mock_processing(prompt: str, seed: str, width: int, height: int, steps: int,
                    cfg_scale: float, sampler: str, batch_count: int):
    args_and_names = {
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": str(cfg_scale),
        "sampler": sampler,
    }

    full_string = f"{prompt}\n" + " ".join([f"{k}:" for k, v in args_and_names.items()])
    info = {
        'text': full_string,
        'entities': [
            {'entity': str(v), 'start': full_string.find(f"{k}:"), 'end': full_string.find(f"{k}:") + len(f"{k} ")} for
            k, v in args_and_names.items()]
    }
    images = []
    for i in range(batch_count):
        images.append(f"http://placeimg.com/{width}/{height}/any")
    return images, int(time.time()), info, 'random output'


def txt2img(*args, **kwargs):
    # Output should match output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats
    # info = f"""{args[0]} --seed {args[9]} --W {args[11]} --H {args[10]} -s {args[1]} -C {float(args[8])} --sampler {args[2]}  """.strip()
    return mock_processing(
        prompt=args[0],
        seed=args[9],
        width=args[11],
        height=args[10],
        steps=args[1],
        cfg_scale=args[8],
        sampler=args[2],
        batch_count=args[6]
    )


def img2img(*args, **kwargs):
    return mock_processing(
        prompt=args[0],
        seed=args[12],
        width=args[14],
        height=args[13],
        steps=args[5],
        cfg_scale=args[10],
        sampler=args[6],
        batch_count=args[9]
    )


def run_GFPGAN(*args, **kwargs):
    time.sleep(.1)
    return "yo"


def run_RealESRGAN(*args, **kwargs):
    time.sleep(.2)
    return "yo"


class model():
    def __init__():
        pass


class opt():
    def __init__(self, name):
        self.name = name

    no_progressbar_hiding = True


css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

css = css_hide_progressbar
css = css + """
[data-testid="image"] {min-height: 512px !important};
#main_body {display:none !important};
#main_body>.col:nth-child(2){width:200%;}
"""

user_defaults = {}

# make sure these indicies line up at the top of txt2img()
txt2img_toggles = [
    'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
    'Normalize Prompt Weights (ensure sum of weights add up to 1.0)',
    'Save individual images',
    'Save grid',
    'Sort samples by prompt',
    'Write sample info files',
    'jpg samples',
]
if GFPGAN is not None:
    txt2img_toggles.append('Fix faces using GFPGAN')
if RealESRGAN is not None:
    txt2img_toggles.append('Upscale images using RealESRGAN')

txt2img_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [1, 2, 3],
    'sampler_name': 'k_lms',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'seed': '',
    'height': 512,
    'width': 512,
    'fp': None,
    'submit_on_enter': 'Yes',
    'variant_amount': 0,
    'variant_seed': ''
}

if 'txt2img' in user_defaults:
    txt2img_defaults.update(user_defaults['txt2img'])

txt2img_toggle_defaults = [txt2img_toggles[i] for i in txt2img_defaults['toggles']]

sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

imgproc_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'sampler_name': 'k_lms',
    'cfg_scale': 7.5,
    'seed': '',
    'height': 512,
    'width': 512,
    'denoising_strength': 0.30
}
imgproc_mode_toggles = [
    'Fix Faces',
    'Upscale'
]

# make sure these indicies line up at the top of img2img()
img2img_toggles = [
    'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
    'Normalize Prompt Weights (ensure sum of weights add up to 1.0)',
    'Loopback (use images from previous batch when creating next batch)',
    'Random loopback seed',
    'Save individual images',
    'Save grid',
    'Sort samples by prompt',
    'Write sample info files',
    'jpg samples',
]
if GFPGAN is not None:
    img2img_toggles.append('Fix faces using GFPGAN')
if RealESRGAN is not None:
    img2img_toggles.append('Upscale images using RealESRGAN')

img2img_mask_modes = [
    "Keep masked area",
    "Regenerate only masked area",
]

img2img_resize_modes = [
    "Just resize",
    "Crop and resize",
    "Resize and fill",
]

img2img_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [1, 4, 5],
    'sampler_name': 'k_lms',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 5.0,
    'denoising_strength': 0.75,
    'mask_mode': 0,
    'resize_mode': 0,
    'seed': '',
    'height': 512,
    'width': 512,
    'fp': None,
}

if 'img2img' in user_defaults:
    img2img_defaults.update(user_defaults['img2img'])

img2img_toggle_defaults = [img2img_toggles[i] for i in img2img_defaults['toggles']]
img2img_image_mode = 'sketch'

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

styling = """
[data-testid="image"] {min-height: 512px !important}
* #body>.col:nth-child(2){width:250%;max-width:89vw}
#generate{width: 100%; }
#prompt_row input{
 font-size:20px
 }
input[type=number]:disabled { -moz-appearance: textfield;+ }
"""

demo = draw_gradio_ui(opt,
                      user_defaults=user_defaults,
                      txt2img=txt2img,
                      img2img=img2img,
                      txt2img_defaults=txt2img_defaults,
                      txt2img_toggles=txt2img_toggles,
                      txt2img_toggle_defaults=txt2img_toggle_defaults,
                      show_embeddings=hasattr(model, "embedding_manager"),
                      img2img_defaults=img2img_defaults,
                      img2img_toggles=img2img_toggles,
                      img2img_toggle_defaults=img2img_toggle_defaults,
                      img2img_mask_modes=img2img_mask_modes,
                      img2img_resize_modes=img2img_resize_modes,
                      sample_img2img=sample_img2img,
                      imgproc_defaults=imgproc_defaults,
                      imgproc_mode_toggles=imgproc_mode_toggles,
                      RealESRGAN=RealESRGAN,
                      GFPGAN=GFPGAN,
                      run_GFPGAN=run_GFPGAN,
                      run_RealESRGAN=run_RealESRGAN
                      )

# demo.queue()
demo.launch(share=False, debug=True)

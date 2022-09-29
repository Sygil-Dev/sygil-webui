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
import re
import gradio as gr
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re


def change_image_editor_mode(choice, cropped_image, masked_image, resize_mode, width, height):
    if choice == "Mask":
        update_image_result = update_image_mask(cropped_image, resize_mode, width, height)
        return [gr.update(visible=False), update_image_result, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]

    update_image_result = update_image_mask(masked_image["image"] if masked_image is not None else None, resize_mode, width, height)
    return [update_image_result, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image, visible=True)

def toggle_options_gfpgan(selection):
    if 0 in selection:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def toggle_options_upscalers(selection):
    if 1 in selection:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def toggle_options_realesrgan(selection):
    if selection == 0 or selection == 1 or selection == 3:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def toggle_options_gobig(selection):
    if selection == 1:
        #print(selection)
        return gr.update(visible=True)
    if selection == 3:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def toggle_options_ldsr(selection):
    if selection == 2 or selection == 3:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def increment_down(value):
    return value - 1

def increment_up(value):
    return value + 1

def copy_img_to_lab(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='imgproc_tab')
        img_update = gr.update(value=processed_image)
        return processed_image, tab_update,
    except IndexError:
        return [None, None]
def copy_img_params_to_lab(params):
    try:
        prompt = params[0][0].replace('\n', ' ').replace('\r', '')
        seed = int(params[1][1])
        steps = int(params[7][1])
        cfg_scale = float(params[9][1])
        sampler = params[11][1]
        return prompt,seed,steps,cfg_scale,sampler
    except IndexError:
        return [None, None]
def copy_img_to_input(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        return processed_image, processed_image , tab_update
    except IndexError:
        return [None, None]

def copy_img_to_edit(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        mode_update = gr.update(value='Crop')
        return processed_image, tab_update, mode_update
    except IndexError:
        return [None, None]

def copy_img_to_mask(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        mode_update = gr.update(value='Mask')
        return processed_image, tab_update, mode_update
    except IndexError:
        return [None, None]



def copy_img_to_upscale_esrgan(img):
    tabs_update = gr.update(selected='realesrgan_tab')
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return processed_image, tabs_update


help_text = """
    ## Mask/Crop
    * Masking is not inpainting. You will probably get better results manually masking your images in photoshop instead.
    * Built-in masking/cropping is very temperamental.
    * It may take some time for the image to show when switching from Crop to Mask.
    * If the image doesn't appear after switching to Mask, switch back to Crop and then back again to Mask
    * If the mask appears distorted (the brush is weirdly shaped instead of round), switch back to Crop and then back again to Mask.

    ## Advanced Editor
    * Click 💾 Save to send your editor changes to the img2img workflow
    * Click ❌ Clear to discard your editor changes

    If anything breaks, try switching modes again, switch tabs, clear the image, or reload.
"""

def resize_image(resize_mode, im, width, height):
    LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res

def update_dimensions_info(width, height):
    pixel_count_formated = "{:,.0f}".format(width * height)
    return f"Aspect ratio: {round(width / height, 5)}\nTotal pixel count: {pixel_count_formated}"

def get_png_nfo( image: Image ):
    info_text = ""
    visible = bool(image and any(image.info))
    if visible:
        for key,value in image.info.items():
            info_text += f"{key}: {value}\n"
        info_text = info_text.rstrip('\n')
    return gr.Textbox.update(value=info_text, visible=visible)

def load_settings(*values):
    new_settings, key_names, checkboxgroup_info = values[-3:]
    values = list(values[:-3])

    if new_settings:
        if type(new_settings) is str:
            if os.path.exists(new_settings):
                with open(new_settings, "r", encoding="utf8") as f:
                    new_settings = yaml.safe_load(f)
            elif new_settings.startswith("file://") and os.path.exists(new_settings[7:]):
                with open(new_settings[7:], "r", encoding="utf8") as f:
                    new_settings = yaml.safe_load(f)
            else:
                new_settings = yaml.safe_load(new_settings)
        if type(new_settings) is not dict:
            new_settings = {"prompt": new_settings}
        if "txt2img" in new_settings:
            new_settings = new_settings["txt2img"]
        target = new_settings.pop("target", "txt2img")
        if target != "txt2img":
            print(f"Warning: applying settings to txt2img even though {target} is specified as target.", file=sys.stderr)

        skipped_settings = {}
        for key in new_settings.keys():
            if key in key_names:
                values[key_names.index(key)] = new_settings[key]
            else:
                skipped_settings[key] = new_settings[key]
        if skipped_settings:
            print(f"Settings could not be applied: {skipped_settings}", file=sys.stderr)

    # Convert lists of checkbox indices to lists of checkbox labels:
    for (cbg_index, cbg_choices) in checkboxgroup_info:
        values[cbg_index] = [cbg_choices[i] for i in values[cbg_index]]

    return values

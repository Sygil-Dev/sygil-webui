import re
import gradio as gr
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re


def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

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

def check_input_for_params(input_text, width, height, steps, seed, number, cfg_scale, sampler):
    # -W, --width
    # -H, --height
    # -s, --steps
    # -S, --seed
    # -n, --number
    # -C, --cfg-scale
    # -A, --sampler

    if height_match := re.search(' (-h|-H|--height)(?P<height> ?\d+) ', input_text):
        height = gr.update(value=int(height_match.group('height')))
        input_text = re.sub(height_match[0], ' ', input_text)
    if width_match := re.search(' (-w|-W|--width)(?P<width> ?\d+) ', input_text):
        width = gr.update(value=int(width_match.group('width')))
        input_text = re.sub(width_match[0], ' ', input_text)
    if steps_match := re.search(' (-s|--steps)(?P<steps> ?\d+) ', input_text):
        steps = gr.update(value=int(steps_match.group('steps')))
        input_text = re.sub(steps_match[0], ' ', input_text)
    if seed_match := re.search(' (-S|--seed)(?P<seed> ?\d+)( |^)', input_text):
        seed = gr.update(value=int(seed_match.group('seed')))
        input_text = re.sub(seed_match[0], ' ', input_text)
    if number_match := re.search(' (-n|-N|--number)(?P<number> ?\d+) ', input_text):
        number = gr.update(value=int(number_match.group('number')))
        input_text = re.sub(number_match[0], ' ', input_text)
    if cfg_scale_match := re.search(' (-c|-C|--cfg-scale)(?P<cfg_scale> ?[\d.?\d]+) ', input_text):
        cfg_scale = gr.update(value=float(cfg_scale_match.group('cfg_scale')))
        input_text = re.sub(cfg_scale_match[0], ' ', input_text)
    if sampler_match := re.search(' (-A|--sampler)(?P<sampler> ?\w+) ', input_text):
        sampler = gr.update(value=sampler_match.group('sampler'))
        input_text = re.sub(sampler_match[0], ' ', input_text)
    return input_text, width, height, steps, seed, number, cfg_scale, sampler
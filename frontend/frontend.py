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
import gradio as gr
from frontend.css_and_js import css, js, call_JS
from frontend.job_manager import JobManager
import frontend.ui_functions as uifn
import uuid
import torch
import os


def draw_gradio_ui(opt, img2img=lambda x: x, txt2img=lambda x: x, imgproc=lambda x: x, scn2img=lambda x: x, 
                   txt2img_defaults={}, RealESRGAN=True, GFPGAN=True, LDSR=True,
                   txt2img_toggles={}, txt2img_toggle_defaults='k_euler', show_embeddings=False, img2img_defaults={},
                   img2img_toggles={}, img2img_toggle_defaults={}, sample_img2img=None, img2img_mask_modes=None,
                   img2img_resize_modes=None, imgproc_defaults={}, imgproc_mode_toggles={}, 
                   scn2img_defaults={}, scn2img_toggles={}, scn2img_toggle_defaults={}, scn2img_define_args=lambda: ({},{},{}),
                   user_defaults={}, run_GFPGAN=lambda x: x, run_RealESRGAN=lambda x: x,
                   job_manager: JobManager = None) -> gr.Blocks:
    with gr.Blocks(css=css(opt), analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
        with gr.Tabs(elem_id='tabss') as tabs:
            with gr.TabItem("Text-to-Image", id='txt2img_tab'):
                with gr.Row(elem_id="prompt_row"):
                    txt2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='prompt_input',
                                                placeholder="A corgi wearing a top hat as an oil painting.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=txt2img_defaults['prompt'],
                                                show_label=False)
                    txt2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(elem_id='body').style(equal_height=False):
                    with gr.Column():
                        txt2img_width = gr.Slider(minimum=64, maximum=1024, step=64, label="Width",
                                                  value=txt2img_defaults["width"])
                        txt2img_height = gr.Slider(minimum=64, maximum=1024, step=64, label="Height",
                                                   value=txt2img_defaults["height"])
                        txt2img_cfg = gr.Slider(minimum=-40.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=txt2img_defaults['cfg_scale'], elem_id='cfg_slider')
                        txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1,
                                                  value=txt2img_defaults["seed"])
                        txt2img_batch_size = gr.Slider(minimum=1, maximum=50, step=1,
                                                       label='Images per batch',
                                                       value=txt2img_defaults['batch_size'])
                        txt2img_batch_count = gr.Slider(minimum=1, maximum=50, step=1,
                                                        label='Number of batches to generate',
                                                        value=txt2img_defaults['n_iter'])

                        txt2img_job_ui = job_manager.draw_gradio_ui() if job_manager else None

                        txt2img_dimensions_info_text_box = gr.Textbox(
                            label="Aspect ratio (4:3 = 1.333 | 16:9 = 1.777 | 21:9 = 2.333)")
                    with gr.Column():
                        with gr.Box():
                            output_txt2img_gallery = gr.Gallery(label="Images", elem_id="txt2img_gallery_output").style(
                                grid=[4, 4])
                            gr.Markdown(
                                "Select an image from the gallery, then click one of the buttons below to perform an action.")
                            with gr.Row(elem_id='txt2img_actions_row'):
                                gr.Button("Copy to clipboard").click(
                                    fn=None,
                                    inputs=output_txt2img_gallery,
                                    outputs=[],
                                    _js=call_JS(
                                        "copyImageFromGalleryToClipboard",
                                        fromId="txt2img_gallery_output"
                                    )
                                )
                                output_txt2img_copy_to_input_btn = gr.Button("Push to img2img")
                                output_txt2img_to_imglab = gr.Button("Send to Lab", visible=True)

                        output_txt2img_params = gr.Highlightedtext(label="Generation parameters", interactive=False,
                                                                   elem_id='txt2img_highlight')
                        with gr.Group():
                            with gr.Row(elem_id='txt2img_output_row'):
                                output_txt2img_copy_params = gr.Button("Copy full parameters").click(
                                    inputs=[output_txt2img_params],
                                    outputs=[],
                                    _js=call_JS(
                                        'copyFullOutput',
                                        fromId='txt2img_highlight'
                                    ),
                                    fn=None, show_progress=False
                                )
                                output_txt2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                output_txt2img_copy_seed = gr.Button("Copy only seed").click(
                                    inputs=[output_txt2img_seed],
                                    outputs=[],
                                    _js=call_JS('gradioInputToClipboard'),
                                    fn=None,
                                    show_progress=False
                                )
                            output_txt2img_stats = gr.HTML(label='Stats')
                    with gr.Column():
                        txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=txt2img_defaults['ddim_steps'])
                        txt2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a',
                                                                'k_euler', 'k_heun', 'k_lms'],
                                                       value=txt2img_defaults['sampler_name'])
                        with gr.Tabs():
                            with gr.TabItem('Simple'):
                                txt2img_submit_on_enter = gr.Radio(['Yes', 'No'],
                                                                   label="Submit on enter? (no means multiline)",
                                                                   value=txt2img_defaults['submit_on_enter'],
                                                                   interactive=True, elem_id='submit_on_enter')
                                txt2img_submit_on_enter.change(
                                    lambda x: gr.update(max_lines=1 if x == 'Yes' else 25), txt2img_submit_on_enter,
                                    txt2img_prompt)
                            with gr.TabItem('Advanced'):
                                txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles,
                                                                   value=txt2img_toggle_defaults, type="index")
                                txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                            choices=['RealESRGAN_x4plus',
                                                                                     'RealESRGAN_x4plus_anime_6B'],
                                                                            value=txt2img_defaults['realesrgan_model_name'],
                                                                       visible=False)  # RealESRGAN is not None # invisible until removed)  # TODO: Feels like I shouldnt slot it in here.

                                txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA",
                                                             value=txt2img_defaults['ddim_eta'], visible=False)
                                txt2img_variant_amount = gr.Slider(minimum=0.0, maximum=1.0, label='Variation Amount',
                                                                   value=txt2img_defaults['variant_amount'])
                                txt2img_variant_seed = gr.Textbox(label="Variant Seed (blank to randomize)", lines=1,
                                                                  max_lines=1, value=txt2img_defaults["variant_seed"])
                        txt2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                txt2img_func = txt2img
                txt2img_inputs = [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles,
                                  txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count,
                                  txt2img_batch_size, txt2img_cfg, txt2img_seed, txt2img_height, txt2img_width,
                                  txt2img_embeddings, txt2img_variant_amount, txt2img_variant_seed]
                txt2img_outputs = [output_txt2img_gallery, output_txt2img_seed,
                                   output_txt2img_params, output_txt2img_stats]

                # If a JobManager was passed in then wrap the Generate functions
                if txt2img_job_ui:
                    txt2img_func, txt2img_inputs, txt2img_outputs = txt2img_job_ui.wrap_func(
                        func=txt2img_func,
                        inputs=txt2img_inputs,
                        outputs=txt2img_outputs
                    )
                    use_queue = False
                else:
                    use_queue = True

                txt2img_btn.click(
                    txt2img_func,
                    txt2img_inputs,
                    txt2img_outputs,
                    api_name='txt2img',
                    queue=use_queue
                )
                txt2img_prompt.submit(
                    txt2img_func,
                    txt2img_inputs,
                    txt2img_outputs,
                    queue=use_queue
                )

                txt2img_width.change(fn=uifn.update_dimensions_info, inputs=[txt2img_width, txt2img_height], outputs=txt2img_dimensions_info_text_box)
                txt2img_height.change(fn=uifn.update_dimensions_info, inputs=[txt2img_width, txt2img_height], outputs=txt2img_dimensions_info_text_box)
                txt2img_dimensions_info_text_box.value = uifn.update_dimensions_info(txt2img_width.value, txt2img_height.value)

                # Temporarily disable prompt parsing until memory issues could be solved
                # See #676
                # live_prompt_params = [txt2img_prompt, txt2img_width, txt2img_height, txt2img_steps, txt2img_seed,
                #                       txt2img_batch_count, txt2img_cfg]
                # txt2img_prompt.change(
                #     fn=None,
                #     inputs=live_prompt_params,
                #     outputs=live_prompt_params,
                #     _js=js_parse_prompt
                # )

            with gr.TabItem("Image-to-Image Unified", id="img2img_tab"):
                with gr.Row(elem_id="prompt_row"):
                    img2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='img2img_prompt_input',
                                                placeholder="A fantasy landscape, trending on artstation.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=img2img_defaults['prompt'],
                                                show_label=False).style()

                    img2img_btn_mask = gr.Button("Generate", variant="primary", visible=False,
                                                 elem_id="img2img_mask_btn")
                    img2img_btn_editor = gr.Button("Generate", variant="primary", elem_id="img2img_edit_btn")
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        gr.Markdown('#### Img2Img Input')
                        img2img_image_mask = gr.Image(
                            value=sample_img2img,
                            source="upload",
                            interactive=True,
                            type="pil", tool="sketch",
                            elem_id="img2img_mask",
                            image_mode="RGBA"
                        )
                        img2img_image_editor = gr.Image(
                            value=sample_img2img,
                            source="upload",
                            interactive=True,
                            type="pil",
                            tool="select",
                            visible=False,
                            image_mode="RGBA",
                            elem_id="img2img_editor"
                        )

                        with gr.Tabs():
                            with gr.TabItem("Editor Options"):
                                with gr.Row():
                                    # disable Uncrop for now
                                    choices=["Mask", "Crop", "Uncrop"]
                                    #choices=["Mask", "Crop"]
                                    img2img_image_editor_mode = gr.Radio(choices=choices,
                                                                         label="Image Editor Mode",
                                                                         value="Mask", elem_id='edit_mode_select',
                                                                         visible=True)
                                    img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"],
                                                            label="Mask Mode", type="index",
                                                            value=img2img_mask_modes[img2img_defaults['mask_mode']],
                                                            visible=True)

                                    img2img_mask_restore = gr.Checkbox(label="Only modify regenerated parts of image",
                                                            value=img2img_defaults['mask_restore'],
                                                            visible=True)

                                    img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=100, step=1,
                                                                           label="How much blurry should the mask be? (to avoid hard edges)",
                                                                           value=img2img_defaults['mask_blur_strength'], visible=True)

                                    img2img_resize = gr.Radio(label="Resize mode",
                                                              choices=["Just resize", "Crop and resize",
                                                                       "Resize and fill"],
                                                              type="index",
                                                              value=img2img_resize_modes[
                                                                  img2img_defaults['resize_mode']], visible=False)

                                img2img_painterro_btn = gr.Button("Advanced Editor")
                            with gr.TabItem("Hints"):
                                img2img_help = gr.Markdown(visible=False, value=uifn.help_text)

                    with gr.Column():
                        gr.Markdown('#### Img2Img Results')
                        output_img2img_gallery = gr.Gallery(label="Images", elem_id="img2img_gallery_output").style(
                            grid=[4, 4, 4])
                        img2img_job_ui = job_manager.draw_gradio_ui() if job_manager else None
                        with gr.Tabs():
                            with gr.TabItem("Generated image actions", id="img2img_actions_tab"):
                                gr.Markdown("Select an image, then press one of the buttons below")
                                with gr.Row():
                                    output_img2img_copy_to_clipboard_btn = gr.Button("Copy to clipboard")
                                    output_img2img_copy_to_input_btn = gr.Button("Push to img2img input")
                                    output_img2img_copy_to_mask_btn = gr.Button("Push to img2img input mask")

                                gr.Markdown("Warning: This will clear your current image and mask settings!")
                            with gr.TabItem("Output info", id="img2img_output_info_tab"):
                                output_img2img_params = gr.Highlightedtext(
                                    label="Generation parameters", interactive=False,
                                    elem_id='img2img_highlight')
                                with gr.Row():
                                    output_img2img_copy_params = gr.Button("Copy full parameters").click(
                                        inputs=output_img2img_params,
                                        outputs=[],
                                        _js=call_JS(
                                            'copyFullOutput',
                                            fromId='img2img_highlight'
                                        ),
                                        fn=None,
                                        show_progress=False)
                                    output_img2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                    output_img2img_copy_seed = gr.Button("Copy only seed").click(
                                        inputs=output_img2img_seed,
                                        outputs=[],
                                        _js=call_JS("gradioInputToClipboard"),
                                        fn=None,
                                        show_progress=False
                                    )
                                output_img2img_stats = gr.HTML(label='Stats')

                gr.Markdown('# img2img settings')

                with gr.Row():
                    with gr.Column():
                        img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                  value=img2img_defaults["width"])
                        img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                   value=img2img_defaults["height"])
                        img2img_cfg = gr.Slider(minimum=-40.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=img2img_defaults['cfg_scale'], elem_id='cfg_slider')
                        img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1,
                                                  value=img2img_defaults["seed"])
                        img2img_batch_count = gr.Slider(minimum=1, maximum=50, step=1,
                                                        label='Batch count (how many batches of images to generate)',
                                                        value=img2img_defaults['n_iter'])
                        img2img_dimensions_info_text_box = gr.Textbox(
                            label="Aspect ratio (4:3 = 1.333 | 16:9 = 1.777 | 21:9 = 2.333)")
                    with gr.Column():
                        img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=img2img_defaults['ddim_steps'])

                        img2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                                'k_heun', 'k_lms'],
                                                       value=img2img_defaults['sampler_name'])

                        img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength',
                                                      value=img2img_defaults['denoising_strength'])

                        img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles,
                                                           value=img2img_toggle_defaults, type="index")

                        img2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                    choices=['RealESRGAN_x4plus',
                                                                             'RealESRGAN_x4plus_anime_6B'],
                                                                    value=img2img_defaults['realesrgan_model_name'],
                                                                    visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.

                        img2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                img2img_image_editor_mode.change(
                    uifn.change_image_editor_mode,
                    [img2img_image_editor_mode,
                     img2img_image_editor,
                     img2img_image_mask,
                     img2img_resize,
                     img2img_width,
                     img2img_height
                     ],
                    [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask,
                     img2img_painterro_btn, img2img_mask, img2img_mask_blur_strength, img2img_mask_restore]
                )

                # img2img_image_editor_mode.change(
                #     uifn.update_image_mask,
                #     [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                #     img2img_image_mask
                # )

                output_txt2img_copy_to_input_btn.click(
                    uifn.copy_img_to_input,
                    [output_txt2img_gallery],
                    [img2img_image_editor, img2img_image_mask, tabs],
                    _js=call_JS("moveImageFromGallery",
                                fromId="txt2img_gallery_output",
                                toId="img2img_editor")
                )

                output_img2img_copy_to_input_btn.click(
                    uifn.copy_img_to_edit,
                    [output_img2img_gallery],
                    [img2img_image_editor, tabs, img2img_image_editor_mode],
                    _js=call_JS("moveImageFromGallery",
                                fromId="img2img_gallery_output",
                                toId="img2img_editor")
                )
                output_img2img_copy_to_mask_btn.click(
                    uifn.copy_img_to_mask,
                    [output_img2img_gallery],
                    [img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=call_JS("moveImageFromGallery",
                                fromId="img2img_gallery_output",
                                toId="img2img_editor")
                )

                output_img2img_copy_to_clipboard_btn.click(fn=None, inputs=output_img2img_gallery, outputs=[],
                                                           _js=call_JS("copyImageFromGalleryToClipboard",
                                                                       fromId="img2img_gallery_output")
                                                           )

                img2img_func = img2img
                img2img_inputs = [img2img_prompt, img2img_image_editor_mode, img2img_mask, img2img_mask_blur_strength,
                                  img2img_mask_restore, img2img_steps, img2img_sampling, img2img_toggles,
                                  img2img_realesrgan_model_name, img2img_batch_count, img2img_cfg,
                                  img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                                  img2img_image_editor, img2img_image_mask, img2img_embeddings]
                img2img_outputs = [output_img2img_gallery, output_img2img_seed, output_img2img_params,
                                   output_img2img_stats]

                # If a JobManager was passed in then wrap the Generate functions
                if img2img_job_ui:
                    img2img_func, img2img_inputs, img2img_outputs = img2img_job_ui.wrap_func(
                        func=img2img_func,
                        inputs=img2img_inputs,
                        outputs=img2img_outputs,
                    )
                    use_queue = False
                else:
                    use_queue = True

                img2img_btn_mask.click(
                    img2img_func,
                    img2img_inputs,
                    img2img_outputs,
                    api_name="img2img",
                    queue=use_queue
                )

                def img2img_submit_params():
                    # print([img2img_prompt, img2img_image_editor_mode, img2img_mask,
                    #              img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                    #              img2img_realesrgan_model_name, img2img_batch_count, img2img_cfg,
                    #              img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                    #              img2img_image_editor, img2img_image_mask, img2img_embeddings])
                    return (img2img_func,
                            img2img_inputs,
                            img2img_outputs)

                img2img_btn_editor.click(*img2img_submit_params())

                # GENERATE ON ENTER
                img2img_prompt.submit(None, None, None,
                                      _js=call_JS("clickFirstVisibleButton",
                                                  rowId="prompt_row"))

                img2img_painterro_btn.click(None,
                                            [img2img_image_editor, img2img_image_mask, img2img_image_editor_mode],
                                            [img2img_image_editor, img2img_image_mask],
                                            _js=call_JS("Painterro.init", toId="img2img_editor")
                                            )

                img2img_width.change(fn=uifn.update_dimensions_info, inputs=[img2img_width, img2img_height],
                                     outputs=img2img_dimensions_info_text_box)
                img2img_height.change(fn=uifn.update_dimensions_info, inputs=[img2img_width, img2img_height],
                                      outputs=img2img_dimensions_info_text_box)
                img2img_dimensions_info_text_box.value = uifn.update_dimensions_info(img2img_width.value, img2img_height.value)

            with gr.TabItem("Image Lab", id='imgproc_tab'):
                gr.Markdown("Post-process results")
                with gr.Row():
                    with gr.Column():
                        with gr.Tabs():
                            with gr.TabItem('Single Image'):
                                imgproc_source = gr.Image(label="Source", source="upload", interactive=True, type="pil",
                                                          elem_id="imglab_input")

                            # gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength",
                            #                            value=gfpgan_defaults['strength'])
                            # select folder with images to process
                            with gr.TabItem('Batch Process'):
                                imgproc_folder = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file")
                        imgproc_pngnfo = gr.Textbox(label="PNG Metadata", placeholder="PngNfo", visible=False,
                                                    max_lines=5)
                        with gr.Row():
                            imgproc_btn = gr.Button("Process", variant="primary")
                        gr.HTML("""
        <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
            <p><b>Upscale Modes Guide</b></p>
            <p></p>
            <p><b>RealESRGAN</b></p>
            <p>A 4X/2X fast upscaler that works well for stylized content, will smooth more detailed compositions.</p>
            <p><b>GoBIG</b></p>
            <p>A 2X upscaler that uses RealESRGAN to upscale the image and then slice it into small parts, each part gets diffused further by SD to create more details, great for adding and increasing details but will change the composition, might also fix issues like eyes etc, use the settings like img2img etc</p>
            <p><b>Latent Diffusion Super Resolution</b></p>
            <p>A 4X upscaler with high VRAM usage that uses a Latent Diffusion model to upscale the image, this will accentuate the details but won't change the composition, might introduce sharpening, great for textures or compositions with plenty of details, is slower.</p>
            <p><b>GoLatent</b></p>
            <p>A 8X upscaler with high VRAM usage, uses GoBig to add details and then uses a Latent Diffusion model to upscale the image, this will result in less artifacting/sharpeninng, use the settings to feed GoBig settings that will contribute to the result, this mode is considerbly slower</p>
        </div>
        """)
                    with gr.Column():
                        with gr.Tabs():
                            with gr.TabItem('Output'):
                                imgproc_output = gr.Gallery(label="Output", elem_id="imgproc_gallery_output")
                        with gr.Row(elem_id="proc_options_row"):
                            with gr.Box():
                                with gr.Column():
                                    gr.Markdown("<b>Processor Selection</b>")
                                    imgproc_toggles = gr.CheckboxGroup(label='', choices=imgproc_mode_toggles,
                                                                       type="index")
                                    # .change toggles to show options
                                    # imgproc_toggles.change()
                        with gr.Box(visible=False) as gfpgan_group:

                            gfpgan_defaults = {
                                'strength': 100,
                            }

                            if 'gfpgan' in user_defaults:
                                gfpgan_defaults.update(user_defaults['gfpgan'])
                            if GFPGAN is None:
                                gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p><b> Please download GFPGAN to activate face fixing features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a></p>
    </div>
    """)
                                # gr.Markdown("")
                                # gr.Markdown("<b> Please download GFPGAN to activate face fixing features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a>")
                            with gr.Column():
                                gr.Markdown("<b>GFPGAN Settings</b>")
                                imgproc_gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001,
                                                                    label="Effect strength",
                                                                    value=gfpgan_defaults['strength'],
                                                                    visible=GFPGAN is not None)
                        with gr.Box(visible=False) as upscale_group:

                            if LDSR:
                                upscaleModes = ['RealESRGAN', 'GoBig', 'Latent Diffusion SR', 'GoLatent ']
                            else:
                                gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p><b> Please download LDSR to activate more upscale features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a></p>
    </div>
    """)
                                upscaleModes = ['RealESRGAN', 'GoBig']
                            with gr.Column():
                                gr.Markdown("<b>Upscaler Selection</b>")
                                imgproc_upscale_toggles = gr.Radio(label='', choices=upscaleModes, type="index",
                                                                   visible=RealESRGAN is not None, value='RealESRGAN')
                        with gr.Box(visible=False) as upscalerSettings_group:

                            with gr.Box(visible=True) as realesrgan_group:
                                with gr.Column():
                                    gr.Markdown("<b>RealESRGAN Settings</b>")
                                    imgproc_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                                interactive=RealESRGAN is not None,
                                                                                choices=['RealESRGAN_x4plus',
                                                                                         'RealESRGAN_x4plus_anime_6B',
                                                                                         'RealESRGAN_x2plus',
                                                                                         'RealESRGAN_x2plus_anime_6B'],
                                                                                value='RealESRGAN_x4plus',
                                                                                visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.
                            with gr.Box(visible=False) as ldsr_group:
                                with gr.Row(elem_id="ldsr_settings_row"):
                                    with gr.Column():
                                        gr.Markdown("<b>Latent Diffusion Super Sampling Settings</b>")
                                        imgproc_ldsr_steps = gr.Slider(minimum=0, maximum=500, step=10,
                                                                       label="LDSR Sampling Steps",
                                                                       value=100, visible=LDSR is not None)
                                        imgproc_ldsr_pre_downSample = gr.Dropdown(
                                            label='LDSR Pre Downsample mode (Lower resolution before processing for speed)',
                                            choices=["None", '1/2', '1/4'], value="None", visible=LDSR is not None)
                                        imgproc_ldsr_post_downSample = gr.Dropdown(
                                            label='LDSR Post Downsample mode (aka SuperSampling)',
                                            choices=["None", "Original Size", '1/2', '1/4'], value="None",
                                            visible=LDSR is not None)
                            with gr.Box(visible=False) as gobig_group:
                                with gr.Row(elem_id="proc_prompt_row"):
                                    with gr.Column():
                                        gr.Markdown("<b>GoBig Settings</b>")
                                        imgproc_prompt = gr.Textbox(label="",
                                                                    elem_id='prompt_input',
                                                                    placeholder="A corgi wearing a top hat as an oil painting.",
                                                                    lines=1,
                                                                    max_lines=1,
                                                                    value=imgproc_defaults['prompt'],
                                                                    show_label=True,
                                                                    visible=RealESRGAN is not None)
                                        imgproc_sampling = gr.Dropdown(
                                            label='Sampling method (k_lms is default k-diffusion sampler)',
                                            choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                     'k_heun', 'k_lms'],
                                            value=imgproc_defaults['sampler_name'], visible=RealESRGAN is not None)
                                        imgproc_steps = gr.Slider(minimum=1, maximum=250, step=1,
                                                                  label="Sampling Steps",
                                                                  value=imgproc_defaults['ddim_steps'],
                                                                  visible=RealESRGAN is not None)
                                        imgproc_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                                value=imgproc_defaults['cfg_scale'],
                                                                visible=RealESRGAN is not None)
                                        imgproc_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                                      label='Denoising Strength',
                                                                      value=imgproc_defaults['denoising_strength'],
                                                                      visible=RealESRGAN is not None)
                                        imgproc_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                                   value=imgproc_defaults["height"],
                                                                   visible=False)  # not currently implemented
                                        imgproc_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                                  value=imgproc_defaults["width"],
                                                                  visible=False)  # not currently implemented
                                        imgproc_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1,
                                                                  max_lines=1,
                                                                  value=imgproc_defaults["seed"],
                                                                  visible=RealESRGAN is not None)
                                        imgproc_btn.click(
                                            imgproc,
                                            [imgproc_source, imgproc_folder, imgproc_prompt, imgproc_toggles,
                                             imgproc_upscale_toggles, imgproc_realesrgan_model_name, imgproc_sampling,
                                             imgproc_steps, imgproc_height,
                                             imgproc_width, imgproc_cfg, imgproc_denoising, imgproc_seed,
                                             imgproc_gfpgan_strength, imgproc_ldsr_steps, imgproc_ldsr_pre_downSample,
                                             imgproc_ldsr_post_downSample],
                                            [imgproc_output], api_name="imgproc")

                                        imgproc_source.change(
                                            uifn.get_png_nfo,
                                            [imgproc_source],
                                            [imgproc_pngnfo])

                                output_txt2img_to_imglab.click(
                                    fn=uifn.copy_img_params_to_lab,
                                    inputs=[output_txt2img_params],
                                    outputs=[imgproc_prompt, imgproc_seed, imgproc_steps, imgproc_cfg,
                                             imgproc_sampling],
                                )

                                output_txt2img_to_imglab.click(
                                    fn=uifn.copy_img_to_lab,
                                    inputs=[output_txt2img_gallery],
                                    outputs=[imgproc_source, tabs],
                                    _js=call_JS("moveImageFromGallery",
                                                fromId="txt2img_gallery_output",
                                                toId="imglab_input")
                                )
                                if RealESRGAN is None:
                                    with gr.Row():
                                        with gr.Column():
                                            # seperator
                                            gr.HTML("""
        <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
            <p><b> Please download RealESRGAN to activate upscale features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a></p>
        </div>
        """)
            imgproc_toggles.change(fn=uifn.toggle_options_gfpgan, inputs=[imgproc_toggles], outputs=[gfpgan_group])
            imgproc_toggles.change(fn=uifn.toggle_options_upscalers, inputs=[imgproc_toggles], outputs=[upscale_group])
            imgproc_toggles.change(fn=uifn.toggle_options_upscalers, inputs=[imgproc_toggles],
                                   outputs=[upscalerSettings_group])
            imgproc_upscale_toggles.change(fn=uifn.toggle_options_realesrgan, inputs=[imgproc_upscale_toggles],
                                           outputs=[realesrgan_group])
            imgproc_upscale_toggles.change(fn=uifn.toggle_options_ldsr, inputs=[imgproc_upscale_toggles],
                                           outputs=[ldsr_group])
            imgproc_upscale_toggles.change(fn=uifn.toggle_options_gobig, inputs=[imgproc_upscale_toggles],
                                           outputs=[gobig_group])

            with gr.TabItem("Scene-to-Image", id='scn2img_tab'):
                example_path = os.path.join("data","scn2img_examples")
                files = os.listdir(example_path)
                examples = {}
                for fn in files:
                    filepath = os.path.join(example_path, str(fn))
                    with open(filepath, "r") as file:
                        examples[fn] = file.read()
                with gr.Row(elem_id="tools_row"):
                    scn2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        scn2img_seed = gr.Textbox(
                            label="Seed (blank to randomize, specify to use cache)", lines=1, max_lines=1,
                            value=scn2img_defaults["seed"]
                        )
                        scn2img_prompt = gr.Textbox(
                            label="Prompt Scene",
                            elem_id='scn2_img_input',
                            placeholder=examples[list(examples.keys())[0]],
                            lines=50,
                            max_lines=50,
                            value=scn2img_defaults['prompt'],
                            show_label=False
                        )

                    with gr.Column():
                        with gr.Tabs():
                            with gr.TabItem("Results", id="scn2img_results_tab"):
                                # gr.Markdown('#### Scn2Img Results')
                                output_scn2img_gallery = gr.Gallery(
                                    label="Images", 
                                    elem_id="scn2img_gallery_output"
                                ).style(grid=[3, 3, 3], height=80)
                                scn2img_job_ui = job_manager.draw_gradio_ui() if job_manager else None
                                
                                with gr.Tabs():
                                    with gr.TabItem("Generated image actions", id="scn2img_actions_tab"):
                                        gr.Markdown("Select an image, then press one of the buttons below")
                                        with gr.Row():
                                            output_scn2img_copy_to_clipboard_btn = gr.Button("Copy to clipboard")
                                            output_scn2img_copy_to_img2img_input_btn = gr.Button("Push to img2img input")
                                            output_scn2img_copy_to_img2img_mask_btn = gr.Button("Push to img2img input mask")

                                        gr.Markdown("Warning: This will clear your current img2img image and mask settings!")

                                    with gr.TabItem("Output info", id="scn2img_output_info_tab"):
                                        output_scn2img_params = gr.Highlightedtext(label="Generation parameters", interactive=False,
                                                                   elem_id='scn2img_highlight')
                                        with gr.Row():
                                            output_scn2img_copy_params = gr.Button("Copy full parameters").click(
                                                inputs=[output_scn2img_params],
                                                outputs=[],
                                                _js=call_JS(
                                                    'copyFullOutput',
                                                    fromId='scn2img_highlight'
                                                ),
                                                fn=None, show_progress=False
                                            )
                                            output_scn2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                            output_scn2img_copy_seed = gr.Button("Copy only initial seed").click(
                                                inputs=output_scn2img_seed, outputs=[],
                                                _js=call_JS("gradioInputToClipboard"), fn=None, show_progress=False)
                                        output_scn2img_stats = gr.HTML(label='Stats')
                                    with gr.TabItem("SceneCode", id="scn2img_scncode_tab"):
                                        output_scn2img_scncode = gr.HTML(label="SceneCode")
                                scn2img_toggles = gr.CheckboxGroup(label='', choices=scn2img_toggles,
                                                                   value=scn2img_toggle_defaults, type="index")

                                scn2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                             visible=show_embeddings)
                            with gr.TabItem("Docs", id="scn2img_docs_tab"):
                                parse_arg, function_args, function_args_ext = scn2img_define_args()
                                with gr.Tabs():
                                    with gr.TabItem("syntax", id=f"scn2img_docs_syntax_tab"):
                                        lines = [
                                            "Scene-to-Image defines layers of images in markdown-like syntax.",
                                            "",
                                            "Markdown headings, e.g. '# layer0', define layers.",
                                            "Layers are hierarchical, i.e. each layer can contain more layers.",
                                            "Child layers are blended together by their image masks, like layers in image editors.",
                                            "",
                                            "The content of sections define the arguments for image generation.",
                                            "Arguments are defined by lines of the form 'arg:value' or 'arg=value'.",
                                            "",
                                            "To invoke txt2img or img2img, they layer must contain the 'prompt' argument.",
                                            "For img2img the layer must have child layers, the result of blending them will be the input image for img2img.",
                                            "When no prompt is specified the layer can still be used for image composition and mask selection.",
                                        ]
                                        gr.Markdown("\n".join(lines))

                                    for func, ext in function_args_ext.items():
                                        with gr.TabItem(func, id=f"scn2img_docs_{func}_tab"):
                                            lines = []
                                            for e in ext:
                                                lines.append(f"#### Arguments for {e}")
                                                if e not in function_args: continue
                                                for argname,argtype in function_args[e].items():
                                                    lines.append(f" - {argname}: {argtype}")
                                            gr.Markdown("\n".join(lines))

                            with gr.TabItem("Examples", id="scn2img_examples_tab"):
                                scn2img_examples = {}
                                with gr.Tabs():
                                    for k, (example, content) in enumerate(examples.items()):
                                        with gr.TabItem(example, id=f"scn2img_example_{k}_tab"):
                                            scn2img_examples[example] = gr.Textbox(
                                                label="Prompt Scene",
                                                elem_id=f"scn2img_example_{k}",
                                                value=content,
                                                lines=50,
                                                max_lines=50,
                                                show_label=False,
                                                interactive=True
                                            )
                output_scn2img_copy_to_img2img_input_btn.click(
                    uifn.copy_img_to_edit,
                    [output_scn2img_gallery],
                    [img2img_image_editor, tabs, img2img_image_editor_mode],
                    _js=call_JS("moveImageFromGallery",
                                fromId="scn2img_gallery_output",
                                toId="img2img_editor")
                )
                output_scn2img_copy_to_img2img_mask_btn.click(
                    uifn.copy_img_to_mask,
                    [output_scn2img_gallery],
                    [img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=call_JS("moveImageFromGallery",
                                fromId="scn2img_gallery_output",
                                toId="img2img_editor")
                )

                output_scn2img_copy_to_clipboard_btn.click(
                    fn=None, 
                    inputs=output_scn2img_gallery, 
                    outputs=[],
                    _js=call_JS("copyImageFromGalleryToClipboard",
                                fromId="scn2img_gallery_output")
                )

                scn2img_func = scn2img
                scn2img_inputs = [
                    scn2img_prompt, 
                    scn2img_toggles,
                    scn2img_seed,
                    scn2img_embeddings
                ]
                scn2img_outputs = [
                    output_scn2img_gallery, 
                    output_scn2img_seed, 
                    output_scn2img_params,
                    output_scn2img_stats,
                    output_scn2img_scncode
                ]
                # If a JobManager was passed in then wrap the Generate functions
                if scn2img_job_ui:
                    scn2img_func, scn2img_inputs, scn2img_outputs = scn2img_job_ui.wrap_func(
                        func=scn2img_func,
                        inputs=scn2img_inputs,
                        outputs=scn2img_outputs,
                    )
                
                scn2img_btn.click(
                    scn2img_func,
                    scn2img_inputs,
                    scn2img_outputs
                )


            """
            if GFPGAN is not None:
                gfpgan_defaults = {
                    'strength': 100,
                }

                if 'gfpgan' in user_defaults:
                    gfpgan_defaults.update(user_defaults['gfpgan'])

                with gr.TabItem("GFPGAN", id='cfpgan_tab'):
                    gr.Markdown("Fix faces on images")
                    with gr.Row():
                        with gr.Column():
                            gfpgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength",
                                                        value=gfpgan_defaults['strength'])
                            gfpgan_btn = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            gfpgan_output = gr.Image(label="Output", elem_id='gan_image')
                    gfpgan_btn.click(
                        run_GFPGAN,
                        [gfpgan_source, gfpgan_strength],
                        [gfpgan_output]
                    )
            if RealESRGAN is not None:
                with gr.TabItem("RealESRGAN", id='realesrgan_tab'):
                    gr.Markdown("Upscale images")
                    with gr.Row():
                        with gr.Column():
                            realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus',
                                                                                                   'RealESRGAN_x4plus_anime_6B'],
                                                                value='RealESRGAN_x4plus')
                            realesrgan_btn = gr.Button("Generate")
                        with gr.Column():
                            realesrgan_output = gr.Image(label="Output", elem_id='gan_image')
                    realesrgan_btn.click(
                        run_RealESRGAN,
                        [realesrgan_source, realesrgan_model_name],
                        [realesrgan_output]
                    )
                output_txt2img_to_upscale_esrgan.click(
                    uifn.copy_img_to_upscale_esrgan,
                    output_txt2img_gallery,
                    [realesrgan_source, tabs],
                    _js=js_move_image('txt2img_gallery_output', 'img2img_editor'))
        """
        gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p>For help and advanced usage guides, visit the <a href="https://github.com/hlky/stable-diffusion-webui/wiki" target="_blank">Project Wiki</a></p>
        <p>Stable Diffusion WebUI is an open-source project. You can find the latest stable builds on the <a href="https://github.com/hlky/stable-diffusion" target="_blank">main repository</a>.
        If you would like to contribute to development or test bleeding edge builds, you can visit the <a href="https://github.com/hlky/stable-diffusion-webui" target="_blank">developement repository</a>.</p>
        <p>Device ID {current_device_index}: {current_device_name}<br/>{total_device_count} total devices</p>
    </div>
    """.format(current_device_name=torch.cuda.get_device_name(), current_device_index=torch.cuda.current_device(), total_device_count=torch.cuda.device_count()))
        # Hack: Detect the load event on the frontend
        # Won't be needed in the next version of gradio
        # See the relevant PR: https://github.com/gradio-app/gradio/pull/2108
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js(opt))
        demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
    return demo

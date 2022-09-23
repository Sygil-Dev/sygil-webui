import gradio as gr
from frontend.css_and_js import call_JS
import frontend.ui_functions as uifn

def ui_image_lab(
    tabs: gr.Tabs,
    txt2img_ui: dict,
    RealESRGAN: bool = True,
    imgproc_defaults={},
    imgproc_mode_toggles={},
    user_defaults={},
    imgproc=lambda x: x,
    GFPGAN=True,
    LDSR=True,
) -> dict:

    img_lab_ui = {}

    with gr.TabItem("Image Lab", id='imgproc_tab'):
        gr.Markdown("Post-process results")
        with gr.Row():

            #  Start Column
            with gr.Column():

                # Single or Batch
                with gr.Tabs():
                    with gr.TabItem('Single Image'):
                        imgproc_source = gr.Image(
                            label="Source", source="upload", interactive=True, type="pil", elem_id="imglab_input"
                        )
                    # gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength",
                    #                            value=gfpgan_defaults['strength'])
                    # select folder with images to process

                    with gr.TabItem('Batch Process'):
                        imgproc_folder = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file")

                imgproc_pngnfo = gr.Textbox(label="PNG Metadata", placeholder="File Metadata", visible=True, max_lines=5)

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
    <p>A 8X upscaler with high VRAM usage, uses GoBig to add details and then uses a Latent Diffusion model to upscale the image, this will result in less artifacting/sharpening, use the settings to feed GoBig settings that will contribute to the result, this mode is considerably slower</p>
</div>
""")
            #  End Column
            with gr.Column():

                with gr.Tabs():
                    with gr.TabItem('Output'):
                        imgproc_output = gr.Gallery(label="Output", elem_id="imgproc_gallery_output")

                # Lab Controls
                with gr.Box():
                    gr.Markdown("<b>Processor Selection</b>")
                    imgproc_toggles = gr.CheckboxGroup(
                        label='', choices=imgproc_mode_toggles, type="index",
                    )

                # Fix Faces
                with gr.Box(visible=False) as gfpgan_group:
                    gfpgan_defaults = { 'strength': 100 }
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

                    gr.Markdown("<b>GFPGAN Settings</b>")
                    imgproc_gfpgan_strength = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.001,
                        label="Effect strength",
                        value=gfpgan_defaults['strength'],
                        visible=GFPGAN is not None
                    )

                # Upscaler choice
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

                    gr.Markdown("<b>Upscaler Selection</b>")
                    imgproc_upscale_toggles = gr.Radio(
                        label='',
                        choices=upscaleModes,
                        type="index",
                        visible=RealESRGAN is not None,
                        value='RealESRGAN'
                    )

                with gr.Box(visible=False) as upscalerSettings_group:

                    with gr.Box(visible=True) as realesrgan_group:
                        with gr.Column():
                            gr.Markdown("<b>RealESRGAN Settings</b>")
                            imgproc_realesrgan_model_name = gr.Dropdown(
                                label='RealESRGAN model',
                                interactive=RealESRGAN is not None,
                                choices=[
                                    'RealESRGAN_x4plus',
                                    'RealESRGAN_x4plus_anime_6B',
                                    'RealESRGAN_x2plus',
                                    'RealESRGAN_x2plus_anime_6B'
                                ],
                                value='RealESRGAN_x4plus',
                                visible=RealESRGAN is not None,
                            )  # TODO: Feels like I shouldn't slot it in here.
                    with gr.Box(visible=False) as ldsr_group:
                        with gr.Row(elem_id="ldsr_settings_row"):
                            with gr.Column():
                                gr.Markdown("<b>Latent Diffusion Super Sampling Settings</b>")
                                imgproc_ldsr_steps = gr.Slider(
                                    minimum=0, maximum=500, step=10,
                                    label="LDSR Sampling Steps",
                                    value=100, visible=LDSR is not None
                                )
                                imgproc_ldsr_pre_downSample = gr.Dropdown(
                                    label='LDSR Pre Downsample mode (Lower resolution before processing for speed)',
                                    choices=["None", '1/2', '1/4'], value="None", visible=LDSR is not None,
                                )
                                imgproc_ldsr_post_downSample = gr.Dropdown(
                                    label='LDSR Post Downsample mode (aka SuperSampling)',
                                    choices=["None", "Original Size", '1/2', '1/4'], value="None",
                                    visible=LDSR is not None,
                                )
                    with gr.Box(visible=False) as gobig_group:
                        with gr.Row(elem_id="proc_prompt_row"):
                            with gr.Column():
                                gr.Markdown("<b>GoBig Settings</b>")
                                imgproc_prompt = gr.Textbox(
                                    label="",
                                    elem_id='prompt_input',
                                    placeholder="A corgi wearing a top hat as an oil painting.",
                                    lines=1,
                                    max_lines=1,
                                    value=imgproc_defaults['prompt'],
                                    show_label=True,
                                    visible=RealESRGAN is not None,
                                )
                                imgproc_sampling = gr.Dropdown(
                                    label='Sampling method (k_lms is default k-diffusion sampler)',
                                    choices=[
                                        "DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                        'k_heun', 'k_lms'
                                    ],
                                    value=imgproc_defaults['sampler_name'], visible=RealESRGAN is not None,
                                )
                                imgproc_steps = gr.Slider(
                                    minimum=1, maximum=250, step=1,
                                    label="Sampling Steps",
                                    value=imgproc_defaults['ddim_steps'],
                                    visible=RealESRGAN is not None,
                                )
                                imgproc_cfg = gr.Slider(
                                    minimum=1.0, maximum=30.0, step=0.5,
                                    label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                    value=imgproc_defaults['cfg_scale'],
                                    visible=RealESRGAN is not None,
                                )
                                imgproc_denoising = gr.Slider(
                                    minimum=0.0, maximum=1.0, step=0.01,
                                    label='Denoising Strength',
                                    value=imgproc_defaults['denoising_strength'],
                                    visible=RealESRGAN is not None,
                                )
                                imgproc_height = gr.Slider(
                                    minimum=64, maximum=2048, step=64, label="Height",
                                    value=imgproc_defaults["height"],
                                    visible=False,
                                )  # not currently implemented
                                imgproc_width = gr.Slider(
                                    minimum=64, maximum=2048, step=64, label="Width",
                                    value=imgproc_defaults["width"],
                                    visible=False,
                                )  # not currently implemented
                                imgproc_seed = gr.Textbox(
                                    label="Seed (blank to randomize)", lines=1,
                                    max_lines=1,
                                    value=imgproc_defaults["seed"],
                                    visible=RealESRGAN is not None,
                                )
                                imgproc_btn.click(
                                    imgproc,
                                    [
                                        imgproc_source, imgproc_folder, imgproc_prompt, imgproc_toggles,
                                        imgproc_upscale_toggles, imgproc_realesrgan_model_name, imgproc_sampling,
                                        imgproc_steps, imgproc_height,
                                        imgproc_width, imgproc_cfg, imgproc_denoising, imgproc_seed,
                                        imgproc_gfpgan_strength, imgproc_ldsr_steps, imgproc_ldsr_pre_downSample,
                                        imgproc_ldsr_post_downSample,
                                    ],
                                    [imgproc_output],
                                    api_name="imgproc",
                                )

                                imgproc_source.change(
                                    uifn.get_png_nfo,
                                    [imgproc_source],
                                    [imgproc_pngnfo],
                                )

                        txt2img_ui['to_imglab_btn'].click(
                            fn=uifn.copy_img_params_to_lab,
                            inputs=[txt2img_ui['output_params']],
                            outputs=[
                                imgproc_prompt, imgproc_seed, imgproc_steps, imgproc_cfg, imgproc_sampling,
                            ],
                        )

                        txt2img_ui['to_imglab_btn'].click(
                            fn=uifn.copy_img_to_lab,
                            inputs=[txt2img_ui['gallery']],
                            outputs=[imgproc_source, tabs],
                            _js=call_JS(
                                "moveImageFromGallery",
                                fromId="txt2img_gallery_output",
                                toId="imglab_input",
                            )
                        )
                        if RealESRGAN is None:
                            with gr.Row():
                                with gr.Column():
                                    # separator
                                    gr.HTML("""
<div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
    <p><b> Please download RealESRGAN to activate upscale features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a></p>
</div>
""")

    imgproc_toggles.change(fn=uifn.toggle_options_gfpgan, inputs=[imgproc_toggles], outputs=[gfpgan_group])
    imgproc_toggles.change(fn=uifn.toggle_options_upscalers, inputs=[imgproc_toggles], outputs=[upscale_group])
    imgproc_toggles.change(
        fn=uifn.toggle_options_upscalers,
        inputs=[imgproc_toggles],
        outputs=[upscalerSettings_group]
    )
    imgproc_upscale_toggles.change(
        fn=uifn.toggle_options_realesrgan,
        inputs=[imgproc_upscale_toggles],
        outputs=[realesrgan_group]
    )
    imgproc_upscale_toggles.change(
        fn=uifn.toggle_options_ldsr,
        inputs=[imgproc_upscale_toggles],
        outputs=[ldsr_group]
    )
    imgproc_upscale_toggles.change(
        fn=uifn.toggle_options_gobig,
        inputs=[imgproc_upscale_toggles],
        outputs=[gobig_group]
    )


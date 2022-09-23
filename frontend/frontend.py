import gradio as gr
from frontend.css_and_js import css, js, call_JS, js_parse_prompt, js_copy_txt2img_output
from frontend.job_manager import JobManager
import frontend.ui_functions as uifn
import torch

from .ui_text_to_image import ui_text_to_image
from .ui_image_to_image import ui_image_to_image
from .ui_image_lab import ui_image_lab

def draw_gradio_ui(
    opt,

    # txt2img
    txt2img=lambda x: x,
    txt2img_defaults={},
    txt2img_toggles={},
    txt2img_toggle_defaults='k_euler',
    show_embeddings=False,


    # img2img
    img2img=lambda x: x,
    img2img_defaults={},
    img2img_toggles={}, img2img_toggle_defaults={}, sample_img2img=None, img2img_mask_modes=None,
    img2img_resize_modes=None, imgproc_defaults={}, imgproc_mode_toggles={}, user_defaults={},
    RealESRGAN=True, # imgLab also

    # imgLab
    imgproc=lambda x: x,
    GFPGAN=True,
    LDSR=True,

    # TODO: Unused?
    run_GFPGAN=lambda x: x,
    run_RealESRGAN=lambda x: x,

    # Common
    job_manager: JobManager = None,
) -> gr.Blocks:

    with gr.Blocks(css=css(opt), analytics_enabled=False, title="Stable Diffusion WebUI") as gr_blocks:
        with gr.Tabs(elem_id='tabs') as tabs:

            txt2img_ui = ui_text_to_image(
                txt2img_func=txt2img,
                txt2img_toggles=txt2img_toggles,
                txt2img_toggle_defaults=txt2img_toggle_defaults,
                txt2img_defaults=txt2img_defaults,
                show_embeddings=show_embeddings,
                job_manager=job_manager,
            )

            img2img_ui = ui_image_to_image(
                tabs,
                txt2img_ui = txt2img_ui, # from txt2img
                txt2img_defaults = txt2img_defaults,
                show_embeddings = show_embeddings,
                img2img = img2img,
                img2img_defaults = img2img_defaults,
                img2img_toggles = img2img_toggles,
                img2img_toggle_defaults = img2img_toggle_defaults,
                sample_img2img = sample_img2img,
                img2img_mask_modes = img2img_mask_modes,
                img2img_resize_modes = img2img_resize_modes,
                RealESRGAN = RealESRGAN,
                job_manager = job_manager,
            )

            image_lab_ui = ui_image_lab(
                tabs = tabs,
                txt2img_ui = txt2img_ui,
                RealESRGAN = RealESRGAN,
                imgproc_defaults = imgproc_defaults,
                imgproc_mode_toggles = imgproc_mode_toggles,
                user_defaults = user_defaults,
                imgproc = imgproc,
                GFPGAN = GFPGAN,
                LDSR = LDSR,
            )

        gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p>For help and advanced usage guides, visit the <a href="https://github.com/hlky/stable-diffusion-webui/wiki" target="_blank">Project Wiki</a></p>
        <p>Stable Diffusion WebUI is an open-source project. You can find the latest stable builds on the <a href="https://github.com/hlky/stable-diffusion" target="_blank">main repository</a>.
        If you would like to contribute to development or test bleeding edge builds, you can visit the <a href="https://github.com/hlky/stable-diffusion-webui" target="_blank">development repository</a>.</p>
        <p>Device ID {current_device_index}: {current_device_name}<br/>{total_device_count} total devices</p>
    </div>
    """.format(current_device_name=torch.cuda.get_device_name(), current_device_index=torch.cuda.current_device(), total_device_count=torch.cuda.device_count()))
        # Hack: Detect the load event on the frontend
        # Won't be needed in the next version of gradio
        # See the relevant PR: https://github.com/gradio-app/gradio/pull/2108
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js(opt))
        gr_blocks.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
    return gr_blocks

import gradio as gr
from frontend.css_and_js import call_JS
from frontend.job_manager import JobManager
import frontend.ui_functions as uifn


def ui_image_to_image(
    ui_tabs: gr.Tabs,
    txt2img_ui: dict,
    txt2img_defaults: dict = {},
    show_embeddings: bool = False,
    img2img_func=lambda x: x,
    img2img_defaults: dict = {},
    img2img_toggles: dict = {},
    img2img_toggle_defaults: dict = {},
    sample_img2img=None,
    img2img_mask_modes=None,
    img2img_resize_modes=None,
    RealESRGAN: bool = True,
    job_manager: JobManager = None,
) -> dict:

    img2img_ui = {}

    with gr.TabItem("Image-to-Image Unified", id="img2img_tab"):

        with gr.Row(elem_id="prompt_row"):
            img2img_ui["prompt"] = gr.Textbox(
                label="Prompt",
                elem_id="img2img_prompt_input",
                placeholder="A fantasy landscape, trending on artstation.",
                lines=1,
                max_lines=1 if txt2img_defaults["submit_on_enter"] == "Yes" else 25,
                value=img2img_defaults["prompt"],
                show_label=False,
            ).style()

            img2img_ui["mask_generate"] = gr.Button(
                "Generate",
                variant="primary",
                visible=False,
                elem_id="img2img_mask_generate",
            )
            img2img_ui["edit_generate"] = gr.Button("Generate", variant="primary", elem_id="img2img_edit_generate")

        with gr.Row().style(equal_height=False):

            # Input Column
            with gr.Column():
                with gr.Box(elem_id="img2img_input_container",):
                    gr.Markdown("#### Img2Img Input")

                    img2img_ui["image_mask"] = gr.Image(
                        value=sample_img2img,
                        source="upload",
                        interactive=True,
                        type="pil",
                        tool="sketch",
                        elem_id="img2img_mask",
                        image_mode="RGBA",
                    )

                    img2img_ui["image_editor"] = gr.Image(
                        value=sample_img2img,
                        source="upload",
                        interactive=True,
                        type="pil",
                        tool="select",
                        visible=False,
                        image_mode="RGBA",
                        elem_id="img2img_editor",
                    )

                with gr.Tabs():
                    with gr.TabItem("Editor Options"):
                        with gr.Row():
                            choices = ["Mask", "Crop", "Uncrop"]

                            img2img_ui["image_editor_mode"] = gr.Radio(
                                choices=choices,
                                label="Image Editor Mode",
                                value="Mask",
                                elem_id="edit_mode_select",
                                visible=True,
                            )

                            img2img_ui["mask"] = gr.Radio(
                                choices=["Keep masked area", "Regenerate only masked area"],
                                label="Mask Mode",
                                type="index",
                                value=img2img_mask_modes[img2img_defaults["mask_mode"]],
                                visible=True,
                            )

                            img2img_ui["mask_restore"] = gr.Checkbox(
                                label="Only modify regenerated parts of image",
                                value=img2img_defaults["mask_restore"],
                                visible=True,
                            )

                            img2img_ui["mask_blur_strength"] = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                label="How much blurry should the mask be? (to avoid hard edges)",
                                value=3,
                                visible=True,
                            )

                            img2img_ui["resize"] = gr.Radio(
                                label="Resize mode",
                                choices=["Just resize", "Crop and resize", "Resize and fill"],
                                type="index",
                                value=img2img_resize_modes[img2img_defaults["resize_mode"]],
                                visible=False,
                            )

                        img2img_ui["painterro_btn"] = gr.Button("Advanced Editor")

                    with gr.TabItem("Hints"):
                        img2img_ui["help"] = gr.Markdown(visible=False, value=uifn.help_text)

            with gr.Column():
                gr.Markdown("#### Img2Img Results")
                img2img_ui["output_gallery"] = gr.Gallery(
                    label="Images",
                    elem_id="img2img_output_gallery",
                ).style(grid=[4, 4, 4])
                img2img_ui["job_ui"] = job_manager.draw_gradio_ui() if job_manager else None

                with gr.Tabs():

                    with gr.TabItem("Generated image actions", id="img2img_actions_tab"):
                        gr.Markdown("Select an image, then press one of the buttons below")
                        with gr.Row():
                            img2img_ui["copy_to_clipboard"] = gr.Button("Copy to clipboard")
                            img2img_ui["copy_to_input_btn"] = gr.Button("Push to img2img input")
                            img2img_ui["copy_to_mask_btn"] = gr.Button("Push to img2img input mask")
                        gr.Markdown("Warning: This will clear your current image and mask settings!")

                    with gr.TabItem("Output info", id="img2img_output_info_tab"):
                        img2img_ui["output_params"] = gr.Textbox(label="Generation parameters")
                        with gr.Row():
                            img2img_ui["output_copy_params"] = gr.Button("Copy full parameters").click(
                                inputs=img2img_ui["output_params"],
                                outputs=[],
                                _js='(x) => {navigator.clipboard.writeText(x.replace(": ",":"))}',
                                fn=None,
                                show_progress=False,
                            )
                            img2img_ui["output_seed"] = gr.Number(label="Seed", interactive=False, visible=False)
                            img2img_ui["copy_seed"] = gr.Button("Copy only seed").click(
                                inputs=img2img_ui["output_seed"],
                                outputs=[],
                                _js=call_JS("gradioInputToClipboard"),
                                fn=None,
                                show_progress=False,
                            )
                        img2img_ui["output_stats"] = gr.HTML(label="Stats")

        gr.Markdown("# img2img settings")

        with gr.Row():
            with gr.Column():
                img2img_ui["width"] = gr.Slider(
                    minimum=64, maximum=2048, step=64, label="Width", value=img2img_defaults["width"]
                )
                img2img_ui["height"] = gr.Slider(
                    minimum=64, maximum=2048, step=64, label="Height", value=img2img_defaults["height"]
                )
                img2img_ui["cfg"] = gr.Slider(
                    minimum=-40.0,
                    maximum=30.0,
                    step=0.5,
                    label="Classifier Free Guidance Scale (how strongly the image should follow the prompt)",
                    value=img2img_defaults["cfg_scale"],
                    elem_id="cfg_slider",
                )
                img2img_ui["seed"] = gr.Textbox(
                    label="Seed (blank to randomize)", lines=1, max_lines=1, value=img2img_defaults["seed"]
                )
                img2img_ui["batch_count"] = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    label="Batch count (how many batches of images to generate)",
                    value=img2img_defaults["n_iter"],
                )
                img2img_ui["dimensions"] = gr.Textbox(label="Aspect ratio (4:3 = 1.333 | 16:9 = 1.777 | 21:9 = 2.333)")

            with gr.Column():
                img2img_ui["steps"] = gr.Slider(
                    minimum=1, maximum=250, step=1, label="Sampling Steps", value=img2img_defaults["ddim_steps"]
                )

                img2img_ui["sampling"] = gr.Dropdown(
                    label="Sampling method (k_lms is default k-diffusion sampler)",
                    choices=["DDIM", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"],
                    value=img2img_defaults["sampler_name"],
                )

                img2img_ui["denoising"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    label="Denoising Strength",
                    value=img2img_defaults["denoising_strength"],
                )

                img2img_ui["toggles"] = gr.CheckboxGroup(
                    label="", choices=img2img_toggles, value=img2img_toggle_defaults, type="index"
                )

                img2img_ui["realesrgan_model_name"] = gr.Dropdown(
                    label="RealESRGAN model",
                    choices=["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"],
                    value="RealESRGAN_x4plus",
                    visible=RealESRGAN is not None,
                )  # TODO: Feels like I shouldn't slot it in here.

                img2img_ui["embeddings"] = gr.File(
                    label="Embeddings file for textual inversion", visible=show_embeddings
                )

        img2img_ui["image_editor_mode"].change(
            uifn.change_image_editor_mode,
            [
                img2img_ui["image_editor_mode"],
                img2img_ui["image_editor"],
                img2img_ui["image_mask"],
                img2img_ui["resize"],
                img2img_ui["width"],
                img2img_ui["height"],
            ],
            [
                img2img_ui["image_editor"],
                img2img_ui["image_mask"],
                img2img_ui["edit_generate"],
                img2img_ui["mask_generate"],
                img2img_ui["painterro_btn"],
                img2img_ui["mask"],
                img2img_ui["mask_blur_strength"],
                img2img_ui["mask_restore"],
            ],
        )

        # img2img_image_editor_mode.change(
        #     uifn.update_image_mask,
        #     [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
        #     img2img_image_mask
        # )

        txt2img_ui["to_img2img_btn"].click(
            uifn.copy_img_to_input,
            [txt2img_ui["gallery"]],
            [img2img_ui["image_editor"], img2img_ui["image_mask"], ui_tabs],
            _js=call_JS("moveImageFromGallery", fromId="txt2img_output_gallery", toId="img2img_editor"),
        )

        img2img_ui["copy_to_input_btn"].click(
            uifn.copy_img_to_edit,
            [img2img_ui["output_gallery"]],
            [img2img_ui["image_editor"], ui_tabs, img2img_ui["image_editor_mode"]],
            _js=call_JS("moveImageFromGallery", fromId="img2img_output_gallery", toId="img2img_editor"),
        )
        img2img_ui["copy_to_mask_btn"].click(
            uifn.copy_img_to_mask,
            [img2img_ui["output_gallery"]],
            [img2img_ui["image_mask"], ui_tabs, img2img_ui["image_editor_mode"]],
            _js=call_JS("moveImageFromGallery", fromId="img2img_output_gallery", toId="img2img_editor"),
        )

        img2img_ui["copy_to_clipboard"].click(
            fn=None,
            inputs=img2img_ui["output_gallery"],
            outputs=[],
            _js=call_JS("copyImageFromGalleryToClipboard", fromId="img2img_output_gallery"),
        )

        img2img_ui["inputs"] = [
            img2img_ui["prompt"],
            img2img_ui["image_editor_mode"],
            img2img_ui["mask"],
            img2img_ui["mask_blur_strength"],
            img2img_ui["mask_restore"],
            img2img_ui["steps"],
            img2img_ui["sampling"],
            img2img_ui["toggles"],
            img2img_ui["realesrgan_model_name"],
            img2img_ui["batch_count"],
            img2img_ui["cfg"],
            img2img_ui["denoising"],
            img2img_ui["seed"],
            img2img_ui["height"],
            img2img_ui["width"],
            img2img_ui["resize"],
            img2img_ui["image_editor"],
            img2img_ui["image_mask"],
            img2img_ui["embeddings"],
        ]
        img2img_ui["outputs"] = [
            img2img_ui["output_gallery"],
            img2img_ui["output_seed"],
            img2img_ui["output_params"],
            img2img_ui["output_stats"],
        ]

        # If a JobManager was passed in then wrap the Generate functions
        if img2img_ui["job_ui"]:
            img2img_func, img2img_ui["inputs"], img2img_ui["outputs"] = img2img_ui["job_ui"].wrap_func(
                func=img2img_func,
                inputs=img2img_ui["inputs"],
                outputs=img2img_ui["outputs"],
            )
            use_queue = False
        else:
            use_queue = True

        img2img_ui["mask_generate"].click(
            img2img_func, img2img_ui["inputs"], img2img_ui["outputs"], api_name="img2img", queue=use_queue
        )

        def img2img_submit_params():
            # print(img2img_ui["inputs"])
            return (img2img_func, img2img_ui["inputs"], img2img_ui["outputs"])

        img2img_ui["edit_generate"].click(*img2img_submit_params())

        # GENERATE ON ENTER
        img2img_ui["prompt"].submit(None, None, None, _js=call_JS("clickFirstVisibleButton", rowId="prompt_row"))

        img2img_ui["painterro_btn"].click(
            None,
            [img2img_ui["image_editor"], img2img_ui["image_mask"], img2img_ui["image_editor_mode"]],
            [img2img_ui["image_editor"], img2img_ui["image_mask"]],
            _js=call_JS("Painterro.init", toId="img2img_editor"),
        )

        img2img_ui["width"].change(
            fn=uifn.update_dimensions_info,
            inputs=[img2img_ui["width"], img2img_ui["height"]],
            outputs=img2img_ui["dimensions"],
        )
        img2img_ui["height"].change(
            fn=uifn.update_dimensions_info,
            inputs=[img2img_ui["width"], img2img_ui["height"]],
            outputs=img2img_ui["dimensions"],
        )
        img2img_ui["dimensions"].value = uifn.update_dimensions_info(
            img2img_ui["width"].value, img2img_ui["height"].value
        )

    return img2img_ui

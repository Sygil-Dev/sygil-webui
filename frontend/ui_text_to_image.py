import gradio as gr
from frontend.css_and_js import call_JS, js_copy_txt2img_output
from frontend.job_manager import JobManager
import frontend.ui_functions as uifn


def ui_text_to_image(
    txt2img_func=lambda x: x,
    txt2img_defaults: dict = {},
    txt2img_toggles: dict = {},
    txt2img_toggle_defaults: str = "k_euler",
    show_embeddings: bool = False,
    job_manager: JobManager = None,
) -> dict:

    # Dict for all the UI elements
    txt2img_ui = {
        "toggles": txt2img_toggles,
        "defaults": txt2img_defaults,
    }

    with gr.TabItem("Text-to-Image", id="txt2img_tab"):

        # Prompt Row
        with gr.Row(elem_id="prompt_row"):
            txt2img_ui["prompt"] = gr.Textbox(
                label="Prompt",
                elem_id="prompt_input",
                placeholder="A corgi wearing a top hat as an oil painting.",
                lines=1,
                max_lines=1 if txt2img_defaults["submit_on_enter"] == "Yes" else 25,
                value=txt2img_defaults["prompt"],
                show_label=False,
            )
            txt2img_ui["generate"] = gr.Button("Generate", elem_id="txt2img_generate", variant="primary")

        with gr.Row(elem_id="body").style(equal_height=False):

            # Start Column
            # with gr.Column(elem_id='txt2img_start_column'): # TODO: Gradio >= 3.3.1
            with gr.Column():
                txt2img_ui["width"] = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    step=64,
                    label="Width",
                    value=txt2img_defaults["width"],
                )
                txt2img_ui["height"] = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    step=64,
                    label="Height",
                    value=txt2img_defaults["height"],
                )
                txt2img_ui["cfg"] = gr.Slider(
                    minimum=-40.0,
                    maximum=30.0,
                    step=0.5,
                    label="Classifier Free Guidance Scale (how strongly the image should follow the prompt)",
                    value=txt2img_defaults["cfg_scale"],
                    elem_id="cfg_slider",
                )
                txt2img_ui["seed"] = gr.Textbox(
                    label="Seed (blank to randomize)",
                    lines=1,
                    max_lines=1,
                    value=txt2img_defaults["seed"],
                )
                txt2img_ui["batch_size"] = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    label="Images per batch",
                    value=txt2img_defaults["batch_size"],
                )
                txt2img_ui["batch_count"] = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    label="Number of batches to generate",
                    value=txt2img_defaults["n_iter"],
                )

                txt2img_ui["job_ui"] = job_manager.draw_gradio_ui() if job_manager else None

                txt2img_ui["dimensions_info_text_box"] = gr.Textbox(
                    label="Aspect ratio (4:3 = 1.333 | 16:9 = 1.777 | 21:9 = 2.333)",
                )

            # Output Column
            # with gr.Column(elem_id="txt2img_output_column"): # TODO: Gradio >= 3.3.1
            with gr.Column():

                # Output gallery Box
                with gr.Box():
                    txt2img_ui["gallery"] = gr.Gallery(label="Images", elem_id="txt2img_output_gallery").style(
                        grid=[4, 4],
                    )
                    gr.Markdown(
                        "Select an image from the gallery, then click one of the buttons below to perform an action.",
                    )

                    # Copy output buttons
                    with gr.Row(elem_id="txt2img_actions_row"):
                        txt2img_ui["to_clipboard_btn"] = gr.Button("Copy to clipboard").click(
                            fn=None,
                            inputs=txt2img_ui["gallery"],
                            outputs=[],
                            _js=call_JS("copyImageFromGalleryToClipboard", fromId="txt2img_output_gallery"),
                        )
                        txt2img_ui["to_img2img_btn"] = gr.Button("Send to img2img")
                        txt2img_ui["to_imglab_btn"] = gr.Button("Send to Lab")

                # Generation parameters
                with gr.Box():
                    txt2img_ui["output_params"] = gr.Highlightedtext(
                        label="Generation parameters",
                        interactive=False,
                        elem_id="output_parameters",
                    )

                    with gr.Group(elem_id="seed_box_container"):
                        txt2img_ui["output_seed"] = gr.Number(
                            elem_id="seed_box",
                            label="Seed",
                            interactive=False,
                            # visible=False,
                            show_label=False,
                        )

                    with gr.Row(elem_id="txt2img_output_row"):
                        txt2img_ui["copy_params_btn"] = gr.Button("Copy Full Parameters").click(
                            inputs=[txt2img_ui["output_params"]],
                            outputs=[],
                            _js=js_copy_txt2img_output,
                            fn=None,
                            show_progress=False,
                        )
                        txt2img_ui["copy_seed_btn"] = gr.Button("Copy Seed").click(
                            inputs=[txt2img_ui["output_seed"]],
                            outputs=[],
                            _js="(x) => navigator.clipboard.writeText(x)",
                            fn=None,
                            show_progress=False,
                        )
                    output_txt2img_stats = gr.HTML(label="Stats")

            # End Column
            # with gr.Column(elem_id='txt2img_end_column'): # TODO: Gradio >= 3.3.1
            with gr.Column():
                txt2img_steps = gr.Slider(
                    minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt2img_defaults["ddim_steps"]
                )
                txt2img_sampling = gr.Dropdown(
                    label="Sampling method (k_lms is default k-diffusion sampler)",
                    choices=[
                        "DDIM",
                        "PLMS",
                        "k_dpm_2_a",
                        "k_dpm_2",
                        "k_euler_a",
                        "k_euler",
                        "k_heun",
                        "k_lms",
                    ],
                    value=txt2img_defaults["sampler_name"],
                )

                # Simple / Advanced Section
                with gr.Tabs():
                    with gr.TabItem("Simple"):
                        txt2img_ui["submit_on_enter"] = gr.Radio(
                            ["Yes", "No"],
                            label="Submit on enter? (no means multiline)",
                            value=txt2img_defaults["submit_on_enter"],
                            interactive=True,
                            elem_id="submit_on_enter",
                        )
                        txt2img_ui["submit_on_enter"].change(
                            lambda x: gr.update(max_lines=1 if x == "Yes" else 25),
                            txt2img_ui["submit_on_enter"],
                            txt2img_ui["prompt"],
                        )

                    with gr.TabItem("Advanced"):
                        txt2img_ui["toggles"] = gr.CheckboxGroup(
                            label="",
                            choices=txt2img_ui["toggles"],
                            value=txt2img_toggle_defaults,
                            type="index",
                        )
                        txt2img_realesrgan_model_name = gr.Dropdown(
                            label="RealESRGAN model",
                            choices=["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"],
                            value="RealESRGAN_x4plus",
                            visible=False,
                        )  # RealESRGAN is not None # invisible until removed)  # TODO: Feels like I shouldn't slot it in here.
                        txt2img_ddim_eta = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            label="DDIM ETA",
                            value=txt2img_defaults["ddim_eta"],
                            visible=False,
                        )
                        txt2img_variant_amount = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            label="Variation Amount",
                            value=txt2img_defaults["variant_amount"],
                        )
                        txt2img_variant_seed = gr.Textbox(
                            label="Variant Seed (blank to randomize)",
                            lines=1,
                            max_lines=1,
                            value=txt2img_defaults["variant_seed"],
                        )
                txt2img_embeddings = gr.File(label="Embeddings file for textual inversion", visible=show_embeddings)

        txt2img_ui["inputs"] = [
            txt2img_ui["prompt"],
            txt2img_steps,
            txt2img_sampling,
            txt2img_ui["toggles"],
            txt2img_realesrgan_model_name,
            txt2img_ddim_eta,
            txt2img_ui["batch_count"],
            txt2img_ui["batch_size"],
            txt2img_ui["cfg"],
            txt2img_ui["seed"],
            txt2img_ui["height"],
            txt2img_ui["width"],
            txt2img_embeddings,
            txt2img_variant_amount,
            txt2img_variant_seed,
        ]

        txt2img_ui["outputs"] = [
            txt2img_ui["gallery"],
            txt2img_ui["output_seed"],
            txt2img_ui["output_params"],
            output_txt2img_stats,
        ]

        # If a JobManager was passed in then wrap the Generate functions
        if txt2img_ui["job_ui"]:
            txt2img_func, txt2img_ui["inputs"], txt2img_ui["outputs"] = txt2img_ui["job_ui"].wrap_func(
                func=txt2img_func, inputs=txt2img_ui["inputs"], outputs=txt2img_ui["outputs"]
            )
            use_queue = False
        else:
            use_queue = True

        txt2img_ui["generate"].click(
            txt2img_func, txt2img_ui["inputs"], txt2img_ui["outputs"], api_name="txt2img", queue=use_queue
        )
        txt2img_ui["prompt"].submit(txt2img_func, txt2img_ui["inputs"], txt2img_ui["outputs"], queue=use_queue)
        txt2img_ui["width"].change(
            fn=uifn.update_dimensions_info,
            inputs=[txt2img_ui["width"], txt2img_ui["height"]],
            outputs=txt2img_ui["dimensions_info_text_box"],
        )
        txt2img_ui["height"].change(
            fn=uifn.update_dimensions_info,
            inputs=[txt2img_ui["width"], txt2img_ui["height"]],
            outputs=txt2img_ui["dimensions_info_text_box"],
        )
        txt2img_ui["dimensions_info_text_box"].value = uifn.update_dimensions_info(
            txt2img_ui["width"].value, txt2img_ui["height"].value
        )

        # Temporarily disable prompt parsing until memory issues could be solved
        # See #676
        # live_prompt_params = [
        #     txt2img_ui["prompt"],
        #     txt2img_ui["width"],
        #     txt2img_ui["height"],
        #     txt2img_ui["steps"],
        #     txt2img_ui["seed"],
        #     txt2img_ui["batch_count"],
        #     txt2img_ui["cfg"],
        # ]
        # txt2img_ui["prompt"].change(
        #     fn=None,
        #     inputs=live_prompt_params,
        #     outputs=live_prompt_params,
        #     _js=js_parse_prompt
        # )

    return txt2img_ui

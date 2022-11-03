# <center>Web-based UI for Stable Diffusion</center>

## Created by [Sygil.Dev](https://github.com/sygil-dev)

## [Join us at Sygil.Dev's Discord Server](https://discord.gg/gyXNe4NySY) [![Discord Server](https://user-images.githubusercontent.com/5977640/190528254-9b5b4423-47ee-4f24-b4f9-fd13fba37518.png)](https://discord.gg/gyXNe4NySY)

## Installation instructions for:

- **[Windows](https://sygil-dev.github.io/sygil-webui/docs/1.windows-installation.html)** 
- **[Linux](https://sygil-dev.github.io/sygil-webui/docs/2.linux-installation.html)**

### Want to ask a question or request a feature?

Come to our [Discord Server](https://discord.gg/gyXNe4NySY) or use [Discussions](https://github.com/sygil-dev/sygil-webui/discussions).

## Documentation

[Documentation is located here](https://sygil-dev.github.io/sygil-webui/)

## Want to contribute?

Check the [Contribution Guide](CONTRIBUTING.md)

[Sygil-Dev](https://github.com/Sygil-Dev) main devs:

* ![hlky's avatar](https://avatars.githubusercontent.com/u/106811348?s=40&v=4) [hlky](https://github.com/hlky)
* ![ZeroCool940711's avatar](https://avatars.githubusercontent.com/u/5977640?s=40&v=4)[ZeroCool940711](https://github.com/ZeroCool940711)
* ![codedealer's avatar](https://avatars.githubusercontent.com/u/4258136?s=40&v=4)[codedealer](https://github.com/codedealer)

### Project Features:

* Built-in image enhancers and upscalers, including GFPGAN and realESRGAN

* Generator Preview: See your image as its being made

* Run additional upscaling models on CPU to save VRAM

* Textual inversion: [Reaserch Paper](https://textual-inversion.github.io/) 

* K-Diffusion Samplers: A great collection of samplers to use, including:
  
  - `k_euler`
  - `k_lms`
  - `k_euler_a`
  - `k_dpm_2`
  - `k_dpm_2_a`
  - `k_heun`
  - `PLMS`
  - `DDIM`

* Loopback: Automatically feed the last generated sample back into img2img

* Prompt Weighting & Negative Prompts: Gain more control over your creations

* Selectable GPU usage from Settings tab

* Word Seeds: Use words instead of seed numbers

* Automated Launcher: Activate conda and run Stable Diffusion with a single command

* Lighter on VRAM: 512x512 Text2Image & Image2Image tested working on 4GB (with *optimized* mode enabled in Settings)

* Prompt validation: If your prompt is too long, you will get a warning in the text output field

* Sequential seeds for batches: If you use a seed of 1000 to generate two batches of two images each, four generated images will have seeds: `1000, 1001, 1002, 1003`.

* Prompt matrix: Separate multiple prompts using the `|` character, and the system will produce an image for every combination of them.

* [Gradio] Advanced img2img editor with Mask and crop capabilities

* [Gradio] Mask painting 🖌️: Powerful tool for re-generating only specific parts of an image you want to change (currently Gradio only)

# SD WebUI

An easy way to work with Stable Diffusion right from your browser.

## Streamlit

![](images/streamlit/streamlit-t2i.png)

**Features:**

- Clean UI with an easy to use design, with support for widescreen displays
- *Dynamic live preview* of your generations
- Easily customizable defaults, right from the WebUI's Settings tab
- An integrated gallery to show the generations for a prompt
- *Optimized VRAM* usage for bigger generations or usage on lower end GPUs
- *Text to Video:* Generate video clips from text prompts right from the WebUI (WIP)
- Image to Text: Use [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator) to interrogate an image and get a prompt that you can use to generate a similar image using Stable Diffusion.
- *Concepts Library:* Run custom embeddings others have made via textual inversion.
- Textual Inversion training: Train your own embeddings on any photo you want and use it on your prompt.
- **Currently in development: [Stable Horde](https://stablehorde.net/) integration; ImgLab, batch inputs, & mask editor from Gradio

**Prompt Weights & Negative Prompts:**

To give a token (tag recognized by the AI) a specific or increased weight (emphasis), add `:0.##` to the prompt, where `0.##` is a decimal that will specify the weight of all tokens before the colon.
Ex: `cat:0.30, dog:0.70` or `guy riding a bicycle :0.7, incoming car :0.30`

Negative prompts can be added by using  `###` , after which any tokens will be seen as negative. 
Ex: `cat playing with string ### yarn` will negate `yarn` from the generated image. 

Negatives are a very powerful tool to get rid of contextually similar or related topics, but **be careful when adding them since the AI might see connections you can't**, and end up outputting gibberish

**Tip:* Try using the same seed with different prompt configurations or weight values see how the AI understands them, it can lead to prompts that are more well-tuned and less prone to error.

Please see the [Streamlit Documentation](docs/4.streamlit-interface.md) to learn more.

## Gradio [Legacy]

![](images/gradio/gradio-t2i.png)

**Features:**

- Older UI that is functional and feature complete.
- Has access to all upscaling models, including LSDR.
- Dynamic prompt entry automatically changes your generation settings based on `--params` in a prompt.
- Includes quick and easy ways to send generations to Image2Image or the Image Lab for upscaling.

**Note: the Gradio interface is no longer being actively developed by Sygil.Dev and is only receiving bug fixes.**

Please see the [Gradio Documentation](docs/5.gradio-interface.md) to learn more.

## Image Upscalers

---

### GFPGAN

![](images/GFPGAN.png)

Lets you improve faces in pictures using the GFPGAN model. There is a checkbox in every tab to use GFPGAN at 100%, and also a separate tab that just allows you to use GFPGAN on any picture, with a slider that controls how strong the effect is.

If you want to use GFPGAN to improve generated faces, you need to install it separately.
Download [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth) and put it
into the `/sygil-webui/models/gfpgan` directory. 

### RealESRGAN

![](images/RealESRGAN.png)

Lets you double the resolution of generated images. There is a checkbox in every tab to use RealESRGAN, and you can choose between the regular upscaler and the anime version.
There is also a separate tab for using RealESRGAN on any picture.

Download [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) and [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth).
Put them into the `sygil-webui/models/realesrgan` directory. 

### LSDR

Download **LDSR** [project.yaml](https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1) and [model last.cpkt](https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1). Rename last.ckpt to model.ckpt and place both under `sygil-webui/models/ldsr/`

### GoBig, and GoLatent *(Currently on the Gradio version Only)*

More powerful upscalers that uses a seperate Latent Diffusion model to more cleanly upscale images.

Please see the [Image Enhancers Documentation](docs/6.image_enhancers.md) to learn more.

-----

### *Original Information From The Stable Diffusion Repo:*

# Stable Diffusion

*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>

**CVPR '22 Oral**

which is available on [GitHub](https://github.com/CompVis/latent-diffusion). PDF at [arXiv](https://arxiv.org/abs/2112.10752). Please also visit our [Project page](https://ommer-lab.com/research/latent-diffusion-models/).

[Stable Diffusion](#stable-diffusion-v1) is a latent text-to-image diffusion
model.
Thanks to a generous compute donation from [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), we were able to train a Latent Diffusion Model on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. 
Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487), 
this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts.
With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See [this section](#stable-diffusion-v1) below and the [model card](https://huggingface.co/CompVis/stable-diffusion).

## Stable Diffusion v1

Stable Diffusion v1 refers to a specific configuration of the model
architecture that uses a downsampling-factor 8 autoencoder with an 860M UNet
and CLIP ViT-L/14 text encoder for the diffusion model. The model was pretrained on 256x256 images and 
then finetuned on 512x512 images.

*Note: Stable Diffusion v1 is a general text-to-image diffusion model and therefore mirrors biases and (mis-)conceptions that are present
in its training data. 
Details on the training procedure and data, as well as the intended use of the model can be found in the corresponding [model card](https://huggingface.co/CompVis/stable-diffusion).

## Comments

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
  and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). 
  Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

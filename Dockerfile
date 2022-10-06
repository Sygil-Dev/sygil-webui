# syntax=docker/dockerfile:1.4

FROM alpine:3.16.2 AS weight_downloader
RUN apk add wget


FROM weight_downloader AS weight_downloader-stable_diffusion
RUN mkdir -p /sd/models/ldm/stable-diffusion-v1
RUN wget --progress=dot:giga https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media -O /sd/models/ldm/stable-diffusion-v1/model.ckpt


FROM weight_downloader AS weight_downloader-gfpgan-gfpgan_1_4
RUN mkdir -p /sd/models/gfpgan
RUN wget --progress=dot:giga https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth -O /sd/models/gfpgan/GFPGANv1.4.pth


FROM weight_downloader AS weight_downloader-gfpgan-detection_resnet
RUN mkdir -p /sd/models/gfpgan/weights
RUN wget --progress=dot:giga https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O /sd/models/gfpgan/weights/detection_Resnet50_Final.pth


FROM weight_downloader AS weight_downloader-gfpgan-parsing_parsenet
RUN mkdir -p /sd/models/gfpgan/weights
RUN wget --progress=dot:giga https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O /sd/models/gfpgan/weights/parsing_parsenet.pth


FROM weight_downloader AS weight_downloader-realesrgan-x4plus
RUN mkdir -p /sd/models/realesrgan/
RUN wget --progress=dot:giga https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /sd/models/realesrgan/RealESRGAN_x4plus.pth


FROM weight_downloader AS weight_downloader-realesrgan-x4plus_anime_6b
RUN mkdir -p /sd/models/realesrgan/
RUN wget --progress=dot:giga https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -O /sd/models/realesrgan/RealESRGAN_x4plus_anime_6B.pth


FROM weight_downloader AS weight_downloader-waifu_diffusion
RUN mkdir -p /sd/models/custom/
RUN wget --progress=dot:giga https://huggingface.co/crumb/pruned-waifu-diffusion/resolve/main/model-pruned.ckpt -O /sd/models/custom/waifu-diffusion.ckpt


FROM weight_downloader AS weight_downloader-trinart_stable_diffusion
RUN mkdir -p /sd/models/custom/
RUN wget --progress=dot:giga https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt -O /sd/models/custom/trinart.ckpt


FROM weight_downloader AS weight_downloader-stable_diffusion_concepts_library
RUN mkdir -p /sd/models/sd-concepts-library
# TODO: fetch concepts as zip from https://github.com/sd-webui/sd-concepts-library/archive/refs/heads/main.zip and extract /sd-concepts-library/* into /models/custom/sd-concepts-library/*


FROM weight_downloader AS weight_downloader-blip_model
RUN mkdir -p /sd/models/blip/
RUN wget --progress=dot:giga https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth -O /sd/models/blip/model__base_caption.pth


FROM weight_downloader AS weight_downloader-ldsr
RUN mkdir -p /sd/models/ldsr/
RUN wget --progress=dot:giga https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1 -O /sd/models/ldsr/model.ckpt
RUN wget --progress=dot:giga https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1 -O /sd/models/ldsr/project.yaml


FROM hlky/pytorch:1.12.1-runtime as stable-diffusion-webui

RUN apt-get update && \
    apt-get install -y wget curl git build-essential zip unzip nano openssh-server libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY ./requirements.txt /install/
RUN /opt/conda/bin/python -m pip install -r /install/requirements.txt
RUN /opt/conda/bin/conda clean -ya

WORKDIR /sd
ENV PYTHONPATH=/sd

COPY --from=weight_downloader-stable_diffusion --link /sd/models/ /sd/models/
COPY --from=weight_downloader-gfpgan-gfpgan_1_4 --link /sd/models/ /sd/models/
COPY --from=weight_downloader-gfpgan-parsing_parsenet --link /sd/models/ /sd/models/
COPY --from=weight_downloader-gfpgan-detection_resnet --link /sd/models/ /sd/models/
COPY --from=weight_downloader-realesrgan-x4plus --link /sd/models/ /sd/models/
COPY --from=weight_downloader-realesrgan-x4plus_anime_6b --link /sd/models/ /sd/models/
COPY --from=weight_downloader-waifu_diffusion --link /sd/models/ /sd/models/
COPY --from=weight_downloader-trinart_stable_diffusion --link /sd/models/ /sd/models/
COPY --from=weight_downloader-stable_diffusion_concepts_library --link /sd/models/ /sd/models/
COPY --from=weight_downloader-blip_model --link /sd/models/ /sd/models/
COPY --from=weight_downloader-ldsr --link /sd/models/ /sd/models/

COPY --link ./data/DejaVuSans.ttf /usr/share/fonts/truetype/
COPY --link ./configs/ /sd/configs/
COPY --link ./data/ /sd/data/
COPY --link ./frontend/ /sd/frontend/
COPY --link ./gfpgan/ /sd/gfpgan/
COPY --link ./images/ /sd/images/
COPY --link ./ldm/ /sd/ldm/
COPY --link ./models/ /sd/models/
COPY --link ./scripts/ /sd/scripts/
COPY --link ./.streamlit/ /sd/.streamlit/
COPY --link ./entrypoint.sh /sd/entrypoint.sh

RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

EXPOSE 8501
VOLUME /sd/outputs

ENTRYPOINT /sd/entrypoint.sh

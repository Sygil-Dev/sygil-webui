ARG IMAGE=hlky/sd-webui:base

FROM ${IMAGE}

WORKDIR /workdir

SHELL ["/bin/bash", "-c"]

ENV PYTHONPATH=/sd

EXPOSE 8501
COPY ./stable-diffusion-webui/data/DejaVuSans.ttf /usr/share/fonts/truetype/
COPY ./stable-diffusion-webui/data/ /sd/data/
copy ./stable-diffusion-webui/images/ /sd/images/
copy ./stable-diffusion-webui/scripts/ /sd/scripts/
copy ./stable-diffusion-webui/ldm/ /sd/ldm/
copy ./stable-diffusion-webui/frontend/ /sd/frontend/
copy ./stable-diffusion-webui/configs/ /sd/configs/
copy ./stable-diffusion-webui/.streamlit/ /sd/.streamlit/
COPY ./stable-diffusion-webui/entrypoint.sh /sd/
ENTRYPOINT /sd/entrypoint.sh

RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

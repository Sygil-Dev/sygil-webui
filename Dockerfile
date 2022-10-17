ARG IMAGE=digburn/sd-webui:base

FROM ${IMAGE}

WORKDIR /workdir

SHELL ["/bin/bash", "-c"]

ENV PYTHONPATH=/sd

EXPOSE 7860
COPY ./data/DejaVuSans.ttf /usr/share/fonts/truetype/
COPY ./data/ /sd/data/
COPY ./images/ /sd/images/
COPY ./scripts/ /sd/scripts/
COPY ./ldm/ /sd/ldm/
COPY ./frontend/ /sd/frontend/
COPY ./configs/ /sd/configs/
COPY ./runpod_entrypoint.sh /sd/entrypoint.sh
RUN chmod +x /sd/entrypoint.sh
ENTRYPOINT /sd/entrypoint.sh

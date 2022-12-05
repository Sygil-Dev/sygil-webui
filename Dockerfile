ARG IMAGE=hlky/sd-webui:base

FROM ${IMAGE}

WORKDIR /workdir

SHELL ["/bin/bash", "-c"]

ENV PYTHONPATH=/sd

EXPOSE 8501
COPY ./entrypoint.sh /sd/
COPY ./data/DejaVuSans.ttf /usr/share/fonts/truetype/
COPY ./data/ /sd/data/
copy ./images/ /sd/images/
copy ./scripts/ /sd/scripts/
copy ./ldm/ /sd/ldm/
copy ./frontend/ /sd/frontend/
copy ./configs/ /sd/configs/
copy ./configs/webui/webui_streamlit.yaml /sd/configs/webui/userconfig_streamlit.yaml
copy ./.streamlit/ /sd/.streamlit/
copy ./optimizedSD/ /sd/optimizedSD/
ENTRYPOINT /sd/entrypoint.sh

RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

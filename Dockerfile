# Assumes host environment is AMD64 architecture

# We should use the Pytorch CUDA/GPU-enabled base image. See:  https://hub.docker.com/r/pytorch/pytorch/tags
# FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# Assumes AMD64 host architecture
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /install

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./sd_requirements.txt /install/
RUN pip install -r /install/sd_requirements.txt

COPY ./requirements.txt /install/
RUN pip install -r /install/requirements.txt

COPY ./ext_requirements.txt /install
RUN pip install -r /install/ext_requirements.txt

COPY ./ui_requirements.txt /install/
RUN pip install -r /install/ui_requirements.txt

# Install font for prompt matrix
COPY /data/DejaVuSans.ttf /usr/share/fonts/truetype/

ENV PYTHONPATH=/sd

EXPOSE 7860 8501

COPY ./entrypoint.sh /sd/
ENTRYPOINT /sd/entrypoint.sh


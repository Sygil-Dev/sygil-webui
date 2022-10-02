# This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

# Copyright 2022 sd-webui team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
# Assumes host environment is AMD64 architecture
# ARG TARGETPLATFORM

# This is used to allow building against AMD GPUs
# Annoyingly, you can't IF branch off of, say, TARGETGPU and set
# the Dockerfile's FROM based on that, so we have to have the user
# pass in the entire image path for us.
ARG PYTORCH_IMAGE=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# To build against AMD, use
# --build-arg PYTORCH_IMAGE=rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.12.1

FROM ${PYTORCH_IMAGE}

WORKDIR /install

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y wget git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /install/

RUN pip install -r /install/requirements.txt
# From base image. We need opencv-python-headless so we uninstall here
RUN pip uninstall -y opencv-python && pip install opencv-python-headless==4.6.0.66

# Install font for prompt matrix
COPY /data/DejaVuSans.ttf /usr/share/fonts/truetype/

ENV PYTHONPATH=/sd

COPY ./models /sd/models
COPY ./configs /sd/configs
COPY ./frontend /sd/frontend
COPY ./ldm /sd/ldm
# COPY ./gfpgan/ /sd/
COPY ./optimizedSD /sd/optimizedSD
COPY ./scripts /sd/scripts

EXPOSE 7860 8501

COPY ./entrypoint.sh /sd/
ENTRYPOINT /sd/entrypoint.sh


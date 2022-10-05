# We are currently stuck on 20.04 until ROCm releases a new version.
FROM ubuntu:20.04

# currently accepts ['amd', 'nvidia']
ARG COMPUTE_DEVICE=nvidia

# Runpod gets its own entrypoint
ARG ENTRYPOINT=entrypoint.sh

WORKDIR /install

SHELL ["/bin/bash", "-c"]

RUN apt-get update

# tzdata has a habit of asking for user input during install
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Python Setup
RUN apt-get install -y wget curl git build-essential zip unzip nano openssh-server libgl1 python3.8 python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# ROCm setup
ENV PYTORCH_ROCM_ARCH=gfx900;gfx906;gfx908;gfx90a;gfx1030
RUN if [[ "$COMPUTE_DEVICE" = "amd" ]]; \
  then \
    wget https://repo.radeon.com/amdgpu-install/22.20.3/ubuntu/focal/amdgpu-install_22.20.50203-1_all.deb \
    && apt-get install -y ./amdgpu-install_22.20.50203-1_all.deb && apt-get update -y \
    && amdgpu-install -y --usecase=rocm,lrt,hip,hiplibsdk --no-dkms \
  ; fi

# Pytorch Setup
RUN if [[ "$COMPUTE_DEVICE" = "amd" ]]; \
  then \
    pip install torch torchvision torchaudio \
      --extra-index-url https://download.pytorch.org/whl/rocm5.1.1 \
  ; else \
    pip install torch torchvision torchaudio \
  ; fi


# COPY ./requirements.txt /install/
# RUN /opt/conda/bin/python -m pip install -r /install/requirements.txt
# RUN /opt/conda/bin/conda clean -ya

COPY ./requirements.txt /install/
RUN pip install -r /install/requirements.txt

WORKDIR /workdir

SHELL ["/bin/bash", "-c"]

ENV PYTHONPATH=/sd
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

EXPOSE 8501
COPY ./data/DejaVuSans.ttf /usr/share/fonts/truetype/
COPY ./data/ /sd/data/
COPY ./images/ /sd/images/
COPY ./scripts/ /sd/scripts/
COPY ./ldm/ /sd/ldm/
COPY ./frontend/ /sd/frontend/
COPY ./configs/ /sd/configs/
COPY ./.streamlit/ /sd/.streamlit/

COPY ./${ENTRYPOINT} /sd/entrypoint.sh

ENTRYPOINT /sd/entrypoint.sh

RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

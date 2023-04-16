ARG IMAGE=tukirito/sygil-webui:base

# Use the base image
FROM ${IMAGE}

# Set the working directory
WORKDIR /workdir

# Use the specified shell
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV PYTHONPATH=/sd

# Expose the required port
EXPOSE 8501

# Copy necessary files and directories
COPY ./entrypoint.sh /sd/
COPY ./data/DejaVuSans.ttf /usr/share/fonts/truetype/
COPY ./data /sd/data
COPY ./images /sd/images
COPY ./scripts /sd/scripts
COPY ./ldm /sd/ldm
COPY ./frontend /sd/frontend
COPY ./configs /sd/configs
COPY ./configs/webui/webui_streamlit.yaml /sd/configs/webui/userconfig_streamlit.yaml
COPY ./.streamlit /sd/.streamlit
COPY ./optimizedSD /sd/optimizedSD

# Set the entrypoint
ENTRYPOINT ["/sd/entrypoint.sh"]

# Create .streamlit directory and set up credentials.toml
RUN mkdir -p ~/.streamlit \
    && echo "[general]" > ~/.streamlit/credentials.toml \
    && echo "email = \"\"" >> ~/.streamlit/credentials.toml

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
import os
import os.path as op
import streamlit as st


class Models:
    pwd=os.popen('pwd').read()
    # Stable DiffusionV1.4
    def modelSD():
        if op.exists(f'models/ldm/stable-diffusion-v1/model.ckpt'):
            return st.write(f"Stable Diffusion model already exists !")
        else:
            # For 4GB model
            # os.system('wget  -O models/ldm/stable-diffusion-v1/model.ckpt https://cdn-lfs.huggingface.co/repos/ab/41/ab41ccb635cd5bd124c8eac1b5796b4f64049c9453c4e50d51819468ca69ceb8/fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556?response-content-disposition=attachment%3B%20filename%3D%22model.ckpt%22')
            # os.rename('models/ldm/stable-diffusion-v1/sd-v1-4.ckpt?alt=media','models/ldm/stable-diffusion-v1/model.ckpt')
            # For 7.2GB model
            os.system(
                'curl -o models/ldm/stable-diffusion-v1/model.ckpt -L https://huggingface.co/kaliansh/sdfull/resolve/main/modelfull.ckpt')
            # os.system('wget -O models/ldm/stable-diffusion-v1/model.ckpt https://cdn-lfs.huggingface.co/repos/ab/41/ab41ccb635cd5bd124c8eac1b5796b4f64049c9453c4e50d51819468ca69ceb8/14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a?response-content-disposition=attachment%3B%20filename%3D%22modelfull.ckpt%22')
            # os.rename('models/ldm/stable-diffusion-v1/modelfull.ckpt','models/ldm/stable-diffusion-v1/model.ckpt')
            return st.write(f"Model installed successfully")

    # RealESRGAN_x4plus & RealESRGAN_x4plus_anime_6B
    def realESRGAN():
        if op.exists('src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus.pth') and op.exists('src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth'):
            return st.write(f"RealESRGAN already exists !")
        else:
            os.system('curl -L https://huggingface.co/kaliansh/sdrep/resolve/main/RealESRGAN_x4plus.pth -o src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus.pth')
            os.system('curl -L https://huggingface.co/kaliansh/sdrep/resolve/main/RealESRGAN_x4plus_anime_6B.pth -o src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth')
            return st.write(f"ESRGAN upscaler installed successfully !")

    # GFPGANv1.3
    def GFPGAN():
        if op.exists('src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth'):
            return st.write(f"GFPGAN already exists !")
        else:
            os.system(
                'curl -L https://huggingface.co/kaliansh/sdrep/resolve/main/GFPGANv1.3.pth -o src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth')
            return st.write(f"GFPGAN installed successfully !")

    # Latent Diffusion
    def modelLD():
        if op.exists('src/latent-diffusion'):
            return st.write(f"Latent-Diffusion Model already esists !")
        else:
            os.system(
                'git clone https://github.com/devilismyfriend/latent-diffusion.git')
            os.system('mv latent-diffusion src/latent-diffusion')
            st.write(f"Github Repository cloned !")
            if op.exists('src/latent-diffusion/experiments/pretrained_models/model.ckpt'):
                st.write(f"Laten Diffusion model already exists!")
            else:
                os.mkdir('src/latent-diffusion/experiments')
                os.mkdir('src/latent-diffusion/experiments/pretrained_models')
                os.system(
                    'curl -o src/latent-diffusion/experiments/pretrained_models/project.yaml -L https://huggingface.co/kaliansh/letentDiff/resolve/main/project.yaml')
                # os.rename('src/latent-diffusion/experiments/pretrained_models/index.html?dl=1', 'src/latent-diffusion/experiments/pretrained_models/project.yaml')
                os.system(
                    'wget -O src/latent-diffusion/experiments/pretrained_models/model.ckpt -L https://huggingface.co/kaliansh/letentDiff/resolve/main/model.ckpt')
                # os.rename('src/latent-diffusion/experiments/pretrained_models/index.html?dl=1', 'src/latent-diffusion/experiments/pretrained_models/model.ckpt')
                return st.write(f"Latent Diffusion successfully installed !")

    # Stable Diffusion Conecpt Library
    def SD_conLib():
        if op.exists('models/custom/sd-concepts-library'):
            return st.write(f"Stable Diffusion Concept Library already exists !")
        else:
            os.system(
                'git clone https://github.com/sd-webui/sd-concepts-library models/custom/')
            return st.write("Stable Diffusion Concept Library successfully installed !")

    # Blip Model
    def modelBlip():
        if op.exists('models/custom/blip/model__base_caption.pth'):
            return st.write(f"Blip Model already exists !")
        else:
            # return st.write(f"Blip Model is to be installed !")
            os.mkdir("models/custom/blip")
            os.system("curl -o models/custom/blip/model__base_caption.pth -L https://huggingface.co/kaliansh/sdrep/resolve/main/model__base_caption.pth")
            return st.write(f"Blip model successfully installed")

    # Waifu Diffusion v1.2
    def modelWD():
        if op.exists("models/custom/waifu-diffusion"):
            return st.write(f"Waifu Diffusion Model already exists !")
        else:
            os.system(
                "git clone https://huggingface.co/hakurei/waifu-diffusion models/custom/waifu-diffusion")
            return st.write(f"Waifu Diffusion model successfully installed")

    # Waifu Diffusion v1.2 Pruned
    def modelWDP():
        if op.exists("models/custom/pruned-waifu-diffusion"):
            return st.write(f"Waifu Pruned Model already exists !")
        else:
            os.system(
                "git clone https://huggingface.co/crumb/pruned-waifu-diffusion models/custom/pruned-waifu-diffusion")
            return st.write(f"Waifu Pruned model successfully installed")

    # TrinArt Stable Diffusion v2
    def modelTSD():
        if op.exists("models/custom/trinart_stable_diffusion_v2"):
            return st.write(f"Trinart S.D model already exists!")
        else:
            os.system(
                "git clone https://huggingface.co/naclbit/trinart_stable_diffusion_v2 models/custom/trinart_stable_diffusion_v2")
            return st.write(f"TrinArt successfully installed !")
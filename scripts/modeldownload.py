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

def updateModels():
    if op.exists('models/ldm/stable-diffusion-v1/model.ckpt'):
        pass
    else:
        os.system('wget https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media -o models/ldm/stable-diffusion-v1/model.ckpt')
        # os.rename('models/ldm/stable-diffusion-v1/sd-v1-4.ckpt?alt=media','models/ldm/stable-diffusion-v1/model.ckpt')
        # os.system('wget https://cdn-lfs.huggingface.co/repos/ab/41/ab41ccb635cd5bd124c8eac1b5796b4f64049c9453c4e50d51819468ca69ceb8/14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a?response-content-disposition=attachment%3B%20filename%3D%22modelfull.ckpt%22 -o models/ldm/stable-diffusion-v1/model.ckpt')
        # os.rename('models/ldm/stable-diffusion-v1/modelfull.ckpt','models/ldm/stable-diffusion-v1/model.ckpt')
    
    if op.exists('models/realesrgan/RealESRGAN_x4plus.pth') and op.exists('models/realesrgan/RealESRGAN_x4plus_anime_6B.pth'):
        pass
    else:
        os.system('wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P models/realesrgan')
        os.system('wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P models/realesrgan')
    
    if op.exists('models/gfpgan/GFPGANv1.3.pth'):
        pass 
    else:
        os.system('wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P models/gfpgan')
    
    if op.exists('models/ldsr'):
        pass
    else:
        os.system('git clone https://github.com/devilismyfriend/latent-diffusion.git')
        os.system('mv latent-diffusion models/ldsr')
    
    if op.exists('models/ldsr/model.ckpt'):
        pass
    else:
        os.mkdir('models/ldsr/experiments')
        os.mkdir('models/ldsr')
        os.system('wget https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1 -o models/ldsr/project.yaml')
        # os.rename('models/ldsr/index.html?dl=1', 'models/ldsr/project.yaml')
        os.system('wget https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1 -o models/ldsr/model.ckpt')
        # os.rename('models/ldsr/index.html?dl=1', 'models/ldsr/model.ckpt')
    
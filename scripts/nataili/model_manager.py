import os
import json
import shutil
import zipfile
import requests
import git
import torch
import hashlib
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from transformers import logging

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from ldm.models.blip import blip_decoder
from tqdm import tqdm
import open_clip
import clip

from nataili.util.cache import torch_gc
from nataili.util import logger

logging.set_verbosity_error()

models = json.load(open('./db.json'))
dependencies = json.load(open('./db_dep.json'))
remote_models = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db.json"
remote_dependencies = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db_dep.json"

class ModelManager():
    def __init__(self, hf_auth=None, download=True):
        if download:
            try:
                logger.init("Model Reference", status="Downloading")
                r = requests.get(remote_models)
                self.models = r.json()
                r = requests.get(remote_dependencies)
                self.dependencies = json.load(open('./db_dep.json'))
                logger.init_ok("Model Reference", status="OK")
            except:
                logger.init_err("Model Reference", status="Download Error")
                self.models = json.load(open('./db.json'))
                self.dependencies = json.load(open('./db_dep.json'))
                logger.init_warn("Model Reference", status="Local")
        self.available_models = []
        self.tainted_models = []
        self.available_dependencies = []
        self.loaded_models = {}
        self.hf_auth = None
        self.set_authentication(hf_auth)

    def init(self):
        dependencies_available = []
        for dependency in self.dependencies:
            if self.check_available(self.get_dependency_files(dependency)):
                dependencies_available.append(dependency)
        self.available_dependencies = dependencies_available

        models_available = []
        for model in self.models:
            if self.check_available(self.get_model_files(model)):
                models_available.append(model)
        self.available_models = models_available

        if self.hf_auth is not None:
            if 'username' not in self.hf_auth and 'password' not in self.hf_auth:
                raise ValueError('hf_auth must contain username and password')
            else:
                if self.hf_auth['username'] == '' or self.hf_auth['password'] == '':
                    raise ValueError('hf_auth must contain username and password')
        return True

    def set_authentication(self, hf_auth=None):
        # We do not let No authentication override previously set auth
        if not hf_auth and self.hf_auth:
            return
        self.hf_auth = hf_auth

    def get_model(self, model_name):
        return self.models.get(model_name)
    
    def get_filtered_models(self, **kwargs):
        '''Get all model names.
        Can filter based on metadata of the model reference db
        '''
        filtered_models = self.models
        for keyword in kwargs:
            iterating_models = filtered_models.copy()
            filtered_models = {}
            for model in iterating_models:
                # logger.debug([keyword,iterating_models[model].get(keyword),kwargs[keyword]])
                if iterating_models[model].get(keyword) == kwargs[keyword]:
                    filtered_models[model] = iterating_models[model]
        return filtered_models

    def get_filtered_model_names(self, **kwargs):
        filtered_models = self.get_filtered_models(**kwargs)
        return list(filtered_models.keys())
    
    def get_dependency(self, dependency_name):
        return self.dependencies[dependency_name]

    def get_model_files(self, model_name):
        return self.models[model_name]['config']['files']
    
    def get_dependency_files(self, dependency_name):
        return self.dependencies[dependency_name]['config']['files']
    
    def get_model_download(self, model_name):
        return self.models[model_name]['config']['download']
    
    def get_dependency_download(self, dependency_name):
        return self.dependencies[dependency_name]['config']['download']
    
    def get_available_models(self):
        return self.available_models
    
    def get_available_dependencies(self):
        return self.available_dependencies
    
    def get_loaded_models(self):
        return self.loaded_models
    
    def get_loaded_models_names(self):
        return list(self.loaded_models.keys())
    
    def get_loaded_model(self, model_name):
        return self.loaded_models[model_name]
    
    def unload_model(self, model_name):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            return True
        return False
    
    def unload_all_models(self):
        for model in self.loaded_models:
            del self.loaded_models[model]
        return True
    
    def taint_model(self,model_name):
        '''Marks a model as not valid by remiving it from available_models'''
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models):
        for model in models:
            self.taint_model(model)

    def load_model_from_config(self, model_path='', config_path='', map_location="cpu"):
        config = OmegaConf.load(config_path)
        pl_sd = torch.load(model_path, map_location=map_location)
        if "global_step" in pl_sd:
            logger.info(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model = model.eval()
        del pl_sd, sd, m, u
        return model

    def load_ckpt(self, model_name='', precision='half', gpu_id=0):
        ckpt_path = self.get_model_files(model_name)[0]['path']
        config_path = self.get_model_files(model_name)[1]['path']
        model = self.load_model_from_config(model_path=ckpt_path, config_path=config_path)
        device = torch.device(f"cuda:{gpu_id}")
        model = (model if precision=='full' else model.half()).to(device)
        torch_gc()
        return {'model': model, 'device': device}
    
    def load_realesrgan(self, model_name='', precision='half', gpu_id=0):
        
        RealESRGAN_models = {
                'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            }

        model_path = self.get_model_files(model_name)[0]['path']
        device = torch.device(f"cuda:{gpu_id}")
        model = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[models[model_name]['name']],
                             pre_pad=0, half=True if precision == 'half' else False, device=device)
        return {'model': model, 'device': device}

    def load_gfpgan(self, model_name='', gpu_id=0):
        
        model_path = self.get_model_files(model_name)[0]['path']
        device = torch.device(f"cuda:{gpu_id}")
        model = GFPGANer(model_path=model_path, upscale=1, arch='clean', 
                         channel_multiplier=2, bg_upsampler=None, device=device)
        return {'model': model, 'device': device}

    def load_blip(self, model_name='', precision='half', gpu_id=0, blip_image_eval_size=512, vit='base'):
        # vit = 'base' or 'large'
        model_path = self.get_model_files(model_name)[0]['path']
        device = torch.device(f"cuda:{gpu_id}")
        model = blip_decoder(pretrained=model_path,
                             med_config="configs/blip/med_config.json",
                             image_size=blip_image_eval_size, vit=vit)
        model = model.eval()
        model = (model if precision=='full' else model.half()).to(device)
        return {'model': model, 'device': device}

    def load_open_clip(self, model_name='', precision='half', gpu_id=0):
        pretrained = self.get_model(model_name)['pretrained_name']
        device = torch.device(f"cuda:{gpu_id}")
        model, _, preprocesses = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir='models/clip')
        model = model.eval()
        model = (model if precision=='full' else model.half()).to(device)
        return {'model': model, 'device': device, 'preprocesses': preprocesses}

    def load_clip(self, model_name='', precision='half', gpu_id=0):
        device = torch.device(f"cuda:{gpu_id}")
        model, preprocesses = clip.load(model_name, device=device, download_root='models/clip')
        model = model.eval()
        model = (model if precision=='full' else model.half()).to(device)
        return {'model': model, 'device': device, 'preprocesses': preprocesses}

    def load_model(self, model_name='', precision='half', gpu_id=0):
        if model_name not in self.available_models:
            return False
        if self.models[model_name]['type'] == 'ckpt':
            self.loaded_models[model_name] = self.load_ckpt(model_name, precision, gpu_id)
            return True
        elif self.models[model_name]['type'] == 'realesrgan':
            self.loaded_models[model_name] = self.load_realesrgan(model_name, precision, gpu_id)
            return True
        elif self.models[model_name]['type'] == 'gfpgan':
            self.loaded_models[model_name] = self.load_gfpgan(model_name, gpu_id)
            return True
        elif self.models[model_name]['type'] == 'blip':
            self.loaded_models[model_name] = self.load_blip(model_name, precision, gpu_id, 512, 'base')
            return True
        elif self.models[model_name]['type'] == 'open_clip':
            self.loaded_models[model_name] = self.load_open_clip(model_name, precision, gpu_id)
            return True
        elif self.models[model_name]['type'] == 'clip':
            self.loaded_models[model_name] = self.load_clip(model_name, precision, gpu_id)
            return True
        else:
            return False

    def validate_model(self, model_name):
        files = self.get_model_files(model_name)
        all_ok = True
        for file_details in files:
            if not self.check_file_available(file_details['path']):
                return False
            if not self.validate_file(file_details):
                return False
        return True

    def validate_file(self, file_details):
        if 'md5sum' in file_details:
            file_name = file_details['path']
            logger.debug(f"Getting md5sum of {file_name}")
            with open(file_name, 'rb') as file_to_check:
                file_hash = hashlib.md5()
                while chunk := file_to_check.read(8192):
                    file_hash.update(chunk)
            if file_details['md5sum'] != file_hash.hexdigest():
                return False
        return True

    def check_file_available(self, file_path):
        return os.path.exists(file_path)

    def check_available(self, files):
        available = True
        for file in files:
            if not self.check_file_available(file['path']):
                available = False
        return available

    def download_file(self, url, file_path):
        # make directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pbar_desc = file_path.split('/')[-1]
        r = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            with tqdm(
                # all optional kwargs
                unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                desc=pbar_desc, total=int(r.headers.get('content-length', 0))
            ) as pbar:
                for chunk in r.iter_content(chunk_size=16*1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))                        

    def download_model(self, model_name):
        if model_name in self.available_models:
            logger.info(f"{model_name} is already available.")
            return True
        download = self.get_model_download(model_name)
        files = self.get_model_files(model_name)
        for i in range(len(download)):         
            file_path = f"{download[i]['file_path']}/{download[i]['file_name']}" if 'file_path' in download[i] else files[i]['path']

            if 'file_url' in download[i]:
                download_url = download[i]['file_url']
                if 'hf_auth' in download[i]:
                    username = self.hf_auth['username']
                    password = self.hf_auth['password']
                    download_url = download_url.format(username=username, password=password)
            if 'file_name' in download[i]:
                download_name = download[i]['file_name']
            if 'file_path' in download[i]:
                download_path = download[i]['file_path']

            if 'manual' in download[i]:
                logger.warning(f"The model {model_name} requires manual download from {download_url}. Please place it in {download_path}/{download_name} then press ENTER to continue...")
                input('')
                continue
            # TODO: simplify
            if "file_content" in download[i]:
                file_content = download[i]['file_content']
                logger.info(f"writing {file_content} to {file_path}")
                # make directory download_path
                os.makedirs(download_path, exist_ok=True)
                # write file_content to download_path/download_name
                with open(os.path.join(download_path, download_name), 'w') as f:
                    f.write(file_content)
            elif 'symlink' in download[i]:
                logger.info(f"symlink {file_path} to {download[i]['symlink']}")
                symlink = download[i]['symlink']
                # make directory symlink
                os.makedirs(download_path, exist_ok=True)
                # make symlink from download_path/download_name to symlink
                os.symlink(symlink, os.path.join(download_path, download_name))
            elif 'git' in download[i]:
                logger.info(f"git clone {download_url} to {file_path}")
                # make directory download_path
                os.makedirs(file_path, exist_ok=True)
                git.Git(file_path).clone(download_url)
                if 'post_process' in download[i]:
                    for post_process in download[i]['post_process']:
                        if 'delete' in post_process:
                            # delete folder post_process['delete']
                            logger.info(f"delete {post_process['delete']}")
                            try:
                                shutil.rmtree(post_process['delete'])
                            except PermissionError as e:
                                logger.error(f"[!] Something went wrong while deleting the `{post_process['delete']}`. Please delete it manually.")
                                logger.error("PermissionError: ", e)
            else:
                if not self.check_file_available(file_path) or model_name in self.tainted_models:
                    logger.debug(f'Downloading {download_url} to {file_path}')
                    self.download_file(download_url, file_path)
                    if not self.validate_model(model_name):
                        return False
        if model_name in self.tainted_models:
            self.tainted_models.remove(model_name)
        self.init()
        return True
    
    def download_dependency(self, dependency_name):
        if dependency_name in self.available_dependencies:
            logger.info(f"{dependency_name} is already installed.")
            return True
        download = self.get_dependency_download(dependency_name)
        files = self.get_dependency_files(dependency_name)
        for i in range(len(download)):
            if "git" in download[i]:
                logger.warning("git download not implemented yet")
                break
            
            file_path = files[i]['path']
            if 'file_url' in download[i]:
                download_url = download[i]['file_url']
            if 'file_name' in download[i]:
                download_name = download[i]['file_name']
            if 'file_path' in download[i]:
                download_path = download[i]['file_path']
            logger.debug(download_name)
            if "unzip" in download[i]:
                zip_path = f'temp/{download_name}.zip'
                # os dirname zip_path
                # mkdir temp
                os.makedirs("temp", exist_ok=True)

                self.download_file(download_url, zip_path)
                logger.info(f"unzip {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('temp/')
                # move temp/sd-concepts-library-main/sd-concepts-library to download_path
                logger.info(f"move temp/{download_name}-main/{download_name} to {download_path}")
                shutil.move(f"temp/{download_name}-main/{download_name}", download_path)
                logger.info(f"delete {zip_path}")
                os.remove(zip_path)
                logger.info(f"delete temp/{download_name}-main/")
                shutil.rmtree(f"temp/{download_name}-main")
            else:
                if not self.check_file_available(file_path):
                    logger.init(f'{file_path}', status="Downloading")
                    self.download_file(download_url, file_path)
        self.init()
        return True
    
    def download_all_models(self):
        for model in self.get_filtered_model_names(download_all = True):
            if not self.check_model_available(model):
                logger.init(f"{model}", status="Downloading")
                self.download_model(model)
            else:
                logger.info(f"{model} is already downloaded.")
        return True
    
    def download_all_dependencies(self):
        for dependency in self.dependencies:
            if not self.check_dependency_available(dependency):
                logger.init(f"{dependency}",status="Downloading")
                self.download_dependency(dependency)
            else:
                logger.info(f"{dependency} is already installed.")
        return True
    
    def download_all(self):
        self.download_all_dependencies()
        self.download_all_models()
        return True
    
    def check_all_available(self):
        for model in self.models:
            if not self.check_available(self.get_model_files(model)):
                return False
        for dependency in self.dependencies:
            if not self.check_available(self.get_dependency_files(dependency)):
                return False
        return True
    
    def check_model_available(self, model_name):
        if model_name not in self.models:
            return False
        return self.check_available(self.get_model_files(model_name))
    
    def check_dependency_available(self, dependency_name):
        if dependency_name not in self.dependencies:
            return False
        return self.check_available(self.get_dependency_files(dependency_name))
    
    def check_all_available(self):
        for model in self.models:
            if not self.check_model_available(model):
                return False
        for dependency in self.dependencies:
            if not self.check_dependency_available(dependency):
                return False
        return True

    


    
        

    

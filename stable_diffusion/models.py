import hashlib
from typing import Optional

import httpx
import logging
import textwrap
import torch

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from ldm.util import instantiate_from_config
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm
from stable_diffusion.configs import stable_diffusion_v1, inpainting_big as inpainting_big_config, celeba_256

logging.basicConfig(level=logging.INFO)

__root__ = Path(__file__).parent.parent


@dataclass
class Models:
    storage_dir: Path

    @dataclass
    class ModelMeta:
        """Contains all the model metadata necessary to run the model."""
        name: str
        """Model name."""

        url: str
        """Where to download the model from."""

        hash: str
        """SHA-256 hash of the downloaded model."""

        file: str
        """The actual model file to load. If zipped, this is the file inside of the zip."""

        config: Optional[DictConfig] = None
        """Model config parameters."""

        unzip: bool = False
        """Whether the model should be unzipped."""

        def download_path(self):
            # we need to use a different name for zip files so that [self.file] can point to the actual model.
            if self.unzip:
                return Path(self.name) / 'model.zip'
            else:
                return Path(self.name) / self.file

    def download(self, model: ModelMeta, checksum: bool = True, overwrite=False):
        """Download the model, and extract if necessary."""
        # ensure directory exists
        path = self.storage_dir / model.download_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            logging.info(f'Model {path} already downloaded. Skipping.')
        else:
            logging.info(f'Downloading model {model.name} into {path} from {model.url} ')
            with httpx.stream('GET', model.url, follow_redirects=True) as response:
                total = int(response.headers['Content-Length'])
                with tqdm(desc=model.name, total=total, unit_scale=True, unit_divisor=1024, unit='B') as progress:
                    num_bytes_downloaded = response.num_bytes_downloaded
                    with path.open('wb') as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
                            progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                            num_bytes_downloaded = response.num_bytes_downloaded
        if checksum:
            self.checksum(model)
        if model.unzip:
            with ZipFile(path, 'r') as zipped:
                for file in tqdm(iterable=zipped.namelist(), total=len(zipped.namelist())):
                    if not (self.storage_dir / file).exists() or overwrite:
                        zipped.extract(member=file, path=self.storage_dir)

    def checksum(self, model_meta: ModelMeta):
        """Ensures downloaded file matches SHA-256 hash"""
        with (self.storage_dir / model_meta.download_path()).open('rb') as file:
            sha = hashlib.sha256()
            chunk_size = 1024
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                sha.update(chunk)

        if sha.hexdigest() != model_meta.hash:
            raise ValueError(textwrap.dedent(f"""
                Downloaded file {model_meta.name} from {model_meta.url} does not match stored hash.
                Expected SHA-256 hash: {model_meta.hash}
                Received SHA-256 hash: {sha.hexdigest()}
            """).lstrip())

    def load_model(self, model_meta: ModelMeta) -> nn.Module:
        """Load the model from disk."""
        if model_meta.config is None:
            raise ValueError(f"Model {model_meta.name} doesn't have a config.yaml. "
                             "You probably meant to call a custom load_model() function.")
        checkpoint = torch.load(self.storage_dir / model_meta.name / model_meta.file, map_location='cpu')
        state_dict = checkpoint['state_dict']
        loaded_model = instantiate_from_config(model_meta.config.model)
        missing_keys, unexpected_keys = loaded_model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            logging.debug(textwrap.dedent(f"""
                missing keys:
                {missing_keys}
            """).lstrip())
        if len(unexpected_keys) > 0:
            logging.debug(textwrap.dedent(f"""
                unexpected keys:
                {unexpected_keys}
            """).lstrip())

        return loaded_model

    stable_diffusion_v1 = ModelMeta(
        name='stable-diffusion-v1',
        url='https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl&download=1',
        hash='fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556',
        file='sd-v1-4.ckpt',
        config=stable_diffusion_v1
    )

    gfpgan = ModelMeta(
        name='gfpgan',
        url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        hash='c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70',
        file='GFPGANv1.3.pth'
    )

    inpainting_big = ModelMeta(
        name='inpainting_big',
        url='https://ommer-lab.com/files/latent-diffusion/inpainting_big.zip',
        hash='6ec79e83b90594096f0bff19efb21e1da5ff62ffbd0291a6b3057fce3213ba3a',
        file='inpainting_big.ckpt',
        config=inpainting_big_config
    )

    celeba_256 = ModelMeta(
        name='celeba-256',
        url='https://ommer-lab.com/files/latent-diffusion/celeba.zip',
        hash='30ec16fd5c2504bbadfd4f5aef0ebfd7209305ea7de9ca6cc72b588294abd735',
        file='model.ckpt',
        config=celeba_256,
        unzip=True
    )


if __name__ == '__main__':
    Models(Path('')).download(Models.stable_diffusion_v1, overwrite=True)

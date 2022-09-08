''' Class to store image generation parameters to be stored as metadata in the image'''
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import copy

@dataclass
class ImageMetadata:
    prompt: str = None
    seed: str = None
    width: str = None
    height: str = None
    steps: str = None
    cfg_scale: str = None
    normalize_prompt_weights: str = None
    denoising_strength: str = None
    GFPGAN: str = None

    def as_png_info(self) -> PngInfo:
        info = PngInfo()
        for key, value in self.as_dict().items():
            info.add_text(key, value)
        return info

    def as_dict(self) -> Dict[str, str]:
        return {f"SD:{key}": str(value) for key, value in asdict(self).items() if value is not None}

    @classmethod
    def set_on_image(cls, image: Image, metadata: ImageMetadata) -> None:
        ''' Sets metadata on image, in both text form and as an ImageMetadata object '''
        if metadata:
            image.info = metadata.as_dict()
        else:
            metadata = ImageMetadata()
        image.info["ImageMetadata"] = copy.copy(metadata)

    @classmethod
    def get_from_image(cls, image: Image) -> Optional[ImageMetadata]:
        ''' Gets metadata from an image, first looking for an ImageMetadata,
            then if not found tries to construct one from the info '''
        metadata = image.info.get("ImageMetadata", None)
        if not metadata:
            found_metadata = False
            metadata = ImageMetadata()
            for key, value in image.info.items():
                if key.lower().startswith("sd:"):
                    key = key[3:]
                    if f"{key}" in metadata.__dict__:
                        metadata.__dict__[key] = value
                        found_metadata = True
            if not found_metadata:
                metadata = None
        if not metadata:
            print("Couldn't find metadata on image")
        return metadata

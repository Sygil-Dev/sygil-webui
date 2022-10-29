# Class realesrgan
# Inputs:
#  - model
#  - device
#  - output_dir
#  - output_ext
# outupts:
#  - output_images
import PIL
from torchvision import transforms
import numpy as np
import os
import cv2

from nataili.util.save_sample import save_sample

class realesrgan:
    def __init__(self, model, device, output_dir, output_ext='jpg'):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.output_images = []
    
    def generate(self, input_image):
        # load image
        img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None
        # upscale
        output, _ = self.model.enhance(img)
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            self.output_ext = 'png'
        
        esrgan_sample = output[:,:,::-1]
        esrgan_image = PIL.Image.fromarray(esrgan_sample)
        # append model name to output image name
        filename = os.path.basename(input_image)
        filename = os.path.splitext(filename)[0]
        filename = f'{filename}_esrgan'
        filename_with_ext = f'{filename}.{self.output_ext}'
        output_image = os.path.join(self.output_dir, filename_with_ext)
        save_sample(esrgan_image, filename, self.output_dir, self.output_ext)
        self.output_images.append(output_image)
        return


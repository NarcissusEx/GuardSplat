##########################################################################
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
# @Date: 2025-04-15
# @Description: Simulate visual distortions for GuardSplat.
# This Software is free for non-commercial, research and evaluation use.
##########################################################################

import random
from functools import partial, reduce

import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F, v2

from compressai.zoo import bmshj2018_factorized
from diff_jpeg import diff_jpeg_coding

class GaussianBlurAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 3
        self.sigma = 0.1
    
    def forward(self, image):
        return F.gaussian_blur(image, self.kernel_size)
    
class BrightnessAttack(nn.Module):
    def __init__(self, factor=0.5):
        super().__init__()
        self.brightness_factor_min = 1 - factor
        self.brightness_factor_max = 1 + factor

    def forward(self, image):
        factor = np.random.rand() * (self.brightness_factor_max - self.brightness_factor_min) + self.brightness_factor_min
        return F.adjust_brightness(image, factor)

# this code is borrowed from Diff-JPEG (https://github.com/necla-ml/Diff-JPEG).
def jpeg_coding_cv2(image_rgb, jpeg_quality):
    B, _, _, _ = image_rgb.shape
    image_rgb_jpeg = []
    for index in range(B):
        encode_parameters = (int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality[index].item()))
        _, encoding = cv2.imencode('.jpeg', image_rgb[index].flip(0).permute(1, 2, 0).numpy(), encode_parameters)
        image_rgb_jpeg.append(torch.from_numpy(cv2.imdecode(encoding, 1)).permute(2, 0, 1).flip(0))
    image_rgb_jpeg: Tensor = torch.stack(image_rgb_jpeg, dim=0)
    return image_rgb_jpeg

class JPEGCompressAttack(nn.Module):
    def __init__(self, flag=True):
        super().__init__()
        self.flag = flag
        self.jpeg_quality = 10.

    def forward(self, image):
        image_rgb = image[None] * 255
        jpeg_quality = torch.tensor([self.jpeg_quality])
        if self.flag:
            v_size = [x // 16 * 16 for x in image_rgb.shape[-2:]]
            image_jpeg = torch.nn.functional.interpolate(image_rgb, size=v_size, mode='bilinear', align_corners=True, antialias=True)
            image_jpeg = diff_jpeg_coding(image_rgb=image_jpeg,
                jpeg_quality=jpeg_quality.cuda(),
                ste=True
            )
            return torch.nn.functional.interpolate(image_jpeg, size=image_rgb.shape[-2:], mode='bilinear', align_corners=True, antialias=True).squeeze(0) / 255.
        
        else:
            return jpeg_coding_cv2(image_rgb=image_rgb.cpu(), jpeg_quality=jpeg_quality).squeeze(0).cuda() / 255

class CropAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 0.4 
    
    def forward(self, image):
        cw, ch = [int(self.scale * x) for x in image.shape[-2:]]
        cropped_img = F.center_crop(image, (cw, ch))
        return cropped_img

class RotationAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle = 30

    def forward(self, image):
        rand_num = random.uniform(-1, 1)
        rotated_img = F.rotate(image, self.angle * rand_num)#, expand = True)
        return rotated_img
    
class ResizeAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_ratio_min = 0.75
        self.resize_ratio_max = 1

    def forward(self, image):
        resize_ratio = np.random.rand() * (self.resize_ratio_max - self.resize_ratio_min) + self.resize_ratio_min
        return torch.nn.functional.interpolate(image[None], scale_factor=(resize_ratio, resize_ratio), mode='bilinear').squeeze()

class GaussianNoiseAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = 0.1

    def forward(self, image):
        return image + (self.sigma ** 2) * torch.randn_like(image)

class VAEAttack(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, image):
        out = self.model(image[None])
        out['x_hat'].clamp_(0, 1)
        return out['x_hat'].squeeze()

class VAEBmshj2018Attack(VAEAttack):
    def __init__(self):
        super().__init__()
        self.model = bmshj2018_factorized(quality=3, pretrained=True).eval().cuda()

class Attacker(nn.Module):
    def __init__(self, attack_types, flag=True):
        super().__init__()
        for attack_type in filter(lambda x : x != 'JPEGCompress', attack_types):
            setattr(self, attack_type, globals()[f'{attack_type}Attack']())

        if 'JPEGCompress' in attack_types:
            self.JPEGCompress = JPEGCompressAttack(flag)

    def forward(self, image, attack_type):
        return getattr(self, attack_type)(image)
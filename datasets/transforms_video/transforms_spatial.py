import math
import random
import warnings
from pathlib import Path

import torch
from torchvision.transforms import Compose, RandomApply
from torchvision.transforms._transforms_video import \
    NormalizeVideo as Normalize
from torchvision.transforms._transforms_video import \
    RandomHorizontalFlipVideo as RandomHorizontalFlip
from torchvision.transforms._transforms_video import ToTensorVideo as ToTensor
import albumentations as alb
""" #ToTensor
Convert tensor data type from uint8 to float, divide value by 255.0 and
permute the dimensions of clip tensor
Args:
    clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
Return:
    clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
"""
from .transforms_tensor import ColorJitter, RandomGrayScale

def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    #_is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0

class ToTensorV2(object):
    """Convert image and mask to `torch.Tensor`."""

    #def __init__(self, always_apply=True, p=1.0):
    #    super().__init__(always_apply=always_apply, p=p)
    def __init__(self):
        pass
    #@property
    #def targets(self):
    #    return self.apply#{"image": self.apply, "mask": self.apply_to_mask}
    def __call__(self,clip,clip2):
        return to_tensor(clip),to_tensor(clip2)#torch.from_numpy(mask)#self.apply

    def apply(self, clip, **params):  # skipcq: PYL-W0613
        return to_tensor(clip)#torch.from_numpy(img.transpose(3, 0, 1,2))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

class Resize:
    def __init__(self, size, interpolation_mode='bilinear'):
        self.size = size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        return torch.nn.functional.interpolate(
            clip, size=self.size, mode=self.interpolation_mode,
            align_corners=False,  # Suppress warning
        )


class RawVideoCrop:
    def get_params(self, clip):
        raise NotImplementedError

    def get_size(self, clip):
        height, width, _ = clip.size()[-3:]
        return height, width

    def __call__(self, clip: torch.Tensor):
        i, j, h, w = self.get_params(clip)
        region = clip[..., i: i + h, j: j + w, :]
        return region.contiguous()


class RawVideoRandomCrop(RawVideoCrop):
    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        #0.4, 1.0
        self.scale = scale
        self.ratio = ratio

    def get_params(self, clip):
        # Mostly copied from torchvision.transforms_video.RandomResizedCrop
        height, width = self.get_size(clip)
        area = height * width
        ratio = self.ratio
        scale = self.scale

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            #print('target_area',target_area) # 74413.33003017394
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            #print('log_ratio',log_ratio) #(-0.2876820724517809, 0.28768207245178085)
            aspect_ratio = math.exp(random.uniform(*log_ratio)) 
            #print('aspect_ratio',aspect_ratio) #1.1658218292863778
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            #print('w',w) #211
            #print('h',h) #238
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                #print('i',i)
                #print('j',j)
                return i, j, h, w #new h, w and new upper left ij

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


class RawVideoRandomCrop_Constrained(RawVideoCrop):
    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        #0.4, 1.0
        self.scale = scale
        self.ratio = ratio
        

    def get_params(self, clip):
        # Mostly copied from torchvision.transforms_video.RandomResizedCrop
        height, width = self.get_size(clip)
        area = height * width
        ratio = self.ratio
        scale = self.scale

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            #print('target_area',target_area) # 74413.33003017394
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            #print('log_ratio',log_ratio) #(-0.2876820724517809, 0.28768207245178085)
            aspect_ratio = math.exp(random.uniform(*log_ratio)) 
            #print('aspect_ratio',aspect_ratio) #1.1658218292863778
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            #print('w',w) #211
            #print('h',h) #238
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                #print('i',i)
                #print('j',j)
                return i, j, h, w #new h, w and new upper left ij

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

class RawVideoCenterMaxCrop(RawVideoCrop):
    def __init__(self, ratio=1.):
        self.ratio = ratio

    def get_params(self, clip):
        height, width = self.get_size(clip)
        if width / height > self.ratio:
            h = height
            w = int(round(h * self.ratio))
        else:
            w = width
            h = int(round(w / self.ratio))
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

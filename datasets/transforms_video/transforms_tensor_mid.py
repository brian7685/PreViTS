import copy
import numbers
import random
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import ToTensorVideo as ToTensor
from torchvision.transforms._transforms_video import \
    RandomHorizontalFlipVideo as RandomHorizontalFlip
from . import functional_tensor as F
import numpy as np
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

class RandomGrayScale:
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img: Tensor):
        num_output_channels = img.shape[0]  # 不改变channels
        if random.random() < self.p:
            return F.rgb_to_grayscale(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd: Callable):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img: Tensor):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class GaussianBlur(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float]) -> None:
        super(GaussianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self._padding: Tuple[int, int] = self.compute_zero_padding(kernel_size)
        kernel = F.get_gaussian_kernel2d(kernel_size, sigma)
        kernel = kernel.repeat(3, 1, 1, 1)
        self.register_buffer('kernel', kernel)

    @staticmethod
    def compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # prepare kernel
        # b, c, h, w = x.shape

        # TODO: explore solution when using jit.trace since it raises a warning
        # because the shape is converted to a tensor instead to a int.
        # convolve tensor with gaussian kernel
        # separate 3 channel so that group=3
        x = x.transpose(0, 1)  # [C, T, H, W] -> [T, C, H, W]
        y = torch.nn.functional.conv2d(x, self.kernel, padding=self._padding, stride=1, groups=3)
        return y.transpose(0, 1).contiguous()




class SequentialGPUCollateFn_mask2_mid: #add mask

    def __init__(self, transform=None, target_transform=True, device=torch.device('cuda')):
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __call__(self, batch):
        clips, masks, load_mask, info,label,use_mid, *others = zip(*batch)  # [[video * 2], label] -> [[video * 2]], [label]
        #print('clips',clips.shape)
        #print('mask',mask.shape)
        #print('label',label)
        #print('other',other)
        #print('load_mask_trans',load_mask)
        #print('load_masks',masks)
        #print('info',info[0])

        use_mid_tensor = torch.as_tensor(use_mid, dtype=torch.long)
        use_mid_tensor = use_mid_tensor.cuda(device=self.device, non_blocking=True)

        grad_loss=1
        if self.target_transform:
            label_tensor = torch.as_tensor(label)
            label_tensor = label_tensor.cuda(device=self.device, non_blocking=True)
        else:
            label_tensor = None

        batch_size = len(clips)
        #print('batch_size',batch_size)
        num_clips = len(clips[0])
        clip_list: List[List[Optional[Tensor]]] = [[None for _ in range(batch_size)] for _ in range(num_clips)]

        mask_list: List[List[Optional[Tensor]]] = [[None for _ in range(batch_size)] for _ in range(num_clips)]
        for batch_index, clip in enumerate(clips):  # batch of clip0, clip1
            if info[batch_index]!=0:
                class_name, vid, mid_frame,frame_path_list,data = info[batch_index]
            for clip_index, clip_tensor in enumerate(clip):
                clip_tensor = clip_tensor.cuda(device=self.device, non_blocking=True)
                """
                clip_tensor = to_tensor(clip_tensor)
                clip_tensor = torch.nn.functional.interpolate(
                    clip_tensor, size=224, mode='bilinear',
                    align_corners=False,  # Suppress warning
                )
                """
                transform_m = Compose([
                RandomHorizontalFlip(),
                ])
                #state = torch.get_rng_state()
                #print('state',state,batch_index,clip_index)
                seed = np.random.randint(2147483647)

                
                clip_tensor = self.transform(clip_tensor)
                random.seed(seed) # apply this seed to img tranfsorms
                torch.manual_seed(seed)
                clip_tensor = transform_m(clip_tensor)
                #torch.set_rng_state(state)
                #mask_tensor = mask
                
                #count non zero
                #print('non zero tensor loading',class_name,vid,np.count_nonzero(masks[batch_index][clip_index]))
                mask_tensor = torch.from_numpy(masks[batch_index][clip_index])
                mask_tensor = mask_tensor.cuda(device=self.device, non_blocking=True)
                #transforms_spatial.RandomHorizontalFlip()
                random.seed(seed) # apply this seed to img tranfsorms
                torch.manual_seed(seed)
                
                
                mask_tensor = transform_m(mask_tensor)#self.transform(mask_tensor)
                mask_list[clip_index][batch_index] = mask_tensor
                #print('non zero tensor loading',class_name,vid,torch.count_nonzero(mask_tensor))
                """
                try:
                    #print('clip_tensor success',clip_tensor.shape)
                    clip_tensor = self.transform(clip_tensor)
                except:
                    print('clip_tensor fail',clip_tensor.shape)
                    print('transform error',clip_index,info[batch_index])
                """
                clip_list[clip_index][batch_index] = clip_tensor

        clip_list: List[Tensor] = [torch.stack(x, dim=0) for x in clip_list]
        mask_list: List[Tensor] = [torch.stack(x, dim=0) for x in mask_list]
        """
        if grad_loss==1:
            mask_list: List[List[Optional[Tensor]]] = [[None for _ in range(batch_size)] for _ in range(num_clips)]
            #try: 
            #if 1:  
            for batch_index, mask in enumerate(masks):  # batch of clip0, clip1
                for mask_index, mask_tensor in enumerate(mask):
                    #print('mask_tensor',mask_tensor)
                    mask_tensor = torch.from_numpy(mask_tensor)
                    mask_tensor = mask_tensor.cuda(device=self.device, non_blocking=True)
                    #mask_tensor = self.transform(mask_tensor)
                    mask_list[mask_index][batch_index] = mask_tensor
            #except:
            #    print('masks',masks[0][0].shape)
            mask_list: List[Tensor] = [torch.stack(x, dim=0) for x in mask_list]
        else:
            mask_list: List[List[Optional[Tensor]]] = [[None for _ in range(batch_size)] for _ in range(num_clips)]
        """
        #print('mask_list',mask_list[1])
        return (clip_list, mask_list, load_mask, info,label_tensor, use_mid_tensor, *others)

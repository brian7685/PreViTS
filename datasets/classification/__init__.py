import logging
import multiprocessing as mp
from pathlib import Path

import torch
from pyhocon import ConfigTree
from torch.utils.data import DataLoader, DistributedSampler

from ..transforms_video import (transforms_spatial, transforms_temporal,
                                transforms_tensor, transforms_tensor_mid)
from .video import VideoDataset
from .video import VideoDataset_visual
from .video_tracking_sal_light import VideoDataset_constrained_mask_tracking_sal_light

logger = logging.getLogger(__name__)


def num_valid_samples(num_samples, rank, num_replicas):
    '''Note: depends on the implementation detail of `DistributedSampler`
    '''
    return (num_samples - rank - 1) // num_replicas + 1


class MainProcessCollateWrapper:
    def __init__(self, dataloader: DataLoader, collate_fn, epoch_callback=None):
        self.dataloader = dataloader
        self.collate_fn = collate_fn
        self.epoch_callback = epoch_callback

    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        #print('it',it)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)
        if self.epoch_callback is not None:
            self.epoch_callback(epoch)

    def num_valid_samples(self):
        sampler = self.dataloader.sampler
        return num_valid_samples(len(sampler.dataset), sampler.rank, sampler.num_replicas)


def identity(x):
    return x


class DataLoaderFactoryV6:

    def __init__(self, cfg: ConfigTree, args, final_validate=False, debug=False):
        self.cfg = cfg
        self.args = args
        self.final_validate = final_validate
        self.debug = debug

    def build(self, vid=False, split='train', device=None):
        logger.info('Building Dataset: VID: %s, Split=%s', vid, split)

        ds_name = self.cfg.get_string('dataset.name')

        if vid: #here
            cpu_transform, gpu_transform = self.get_transform_vid()

            size = self.cfg.get_int('temporal_transforms.size')
            strides = self.cfg.get_list('temporal_transforms.strides')
            logger.info(f'Using frames: {size}, with strides: {strides}')
            temporal_transform = transforms_temporal.RandomStrideCrop(
                size=size,
                strides=strides,
            )
        else:
            cpu_transform, gpu_transform = self.get_transform(split)
            temporal_transform = self.get_temporal_transform(split)

        if ds_name == 'ucf101':
            from .ucf101 import UCF101
            ds = UCF101(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'vgg_sound':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_test':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split='test',
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_debug':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'trackingNet':
            from .trackingNet import TrackingNet
            ds = TrackingNet(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'davis':
            from .davis import davis
            ds = davis(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'r2v2':
            from .r2v2 import R2V2
            ds = R2V2(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'kinetics_test':  # [kinetics100, kinetics400]
            from .kinetics import Kinetics_test
            ds = Kinetics_test(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('kinetics'):  # [kinetics100, kinetics400]
            from .kinetics import Kinetics
            ds = Kinetics(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('hmdb51'):
            from .hmdb51 import HMDB51
            ds = HMDB51(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'smth_smth':
            from .smth_smth import Smth_Smth
            ds = Smth_Smth(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                split=split,
            )
        else:
            raise ValueError(f'Unknown dataset "{ds_name}"')
        #gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn(
        
        gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn_mask2(
            transform=gpu_transform,
            target_transform=(not vid),  # VID task does not need labels
            device=device,
        )

        video_dataset = VideoDataset_constrained_mask_tracking_sal_light(
            samples=ds,
            temporal_transform=temporal_transform,
            spatial_transform=cpu_transform,
            num_clips_per_sample=2 if vid else 1,
            frame_rate=self.cfg.get_float('temporal_transforms.frame_rate'),
            args = self.args,
        )

        sampler = DistributedSampler(ds, shuffle=(split == 'train'))

        if split == 'train':
            batch_size = self.cfg.get_int('batch_size')
        elif self.final_validate:
            batch_size = self.cfg.get_int('final_validate.batch_size')
        else:
            batch_size = self.cfg.get_int('validate.batch_size')

        dl = DataLoader(
            video_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.get_int('num_workers'),
            sampler=sampler,
            drop_last=(split == 'train'),
            collate_fn=identity,  # We will deal with collation on main thread.
            multiprocessing_context=mp.get_context('fork'),
        )

        return MainProcessCollateWrapper(dl, gpu_collate_fn)


    def _get_normalize(self):
        if self.debug:
            def normalize(x):
                return x
        else:
            normalize = transforms_spatial.Normalize(
                self.cfg.get_list('dataset.mean'),
                self.cfg.get_list('dataset.std'),
                inplace=True,
            )
        return normalize

    def get_transform_vid(self): #here
        arch = self.cfg.get_string('model.arch')
        aug_plus = self.cfg.get_bool('moco.aug_plus')

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')

        logger.info(f'Training {arch} with size: {size}')

        cpu_transform = transforms_tensor.Compose([
            
            transforms_spatial.RawVideoRandomCrop(
                scale=(
                    0.4, 1.0
                )
            ),

        ])

        normalize = self._get_normalize()

        if not aug_plus: #this one
            
            if int(self.args.with_flip)==1:
                gpu_transform = transforms_tensor.Compose([ #this one
                    transforms_spatial.ToTensor(),
                    transforms_spatial.Resize(size),
                    transforms_spatial.RandomGrayScale(p=0.2),
                    transforms_spatial.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.4,
                    ),
                    transforms_spatial.RandomHorizontalFlip(),
                    normalize,
                ])
            else:
                gpu_transform = transforms_tensor.Compose([ #this one
                    transforms_spatial.ToTensor(),
                    transforms_spatial.Resize(size),
                    transforms_spatial.RandomGrayScale(p=0.2),
                    transforms_spatial.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.4,
                    ),
                    #transforms_spatial.RandomHorizontalFlip(),
                    normalize,
                ])
           
        else: 
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomApply([
                    transforms_spatial.ColorJitter(
                        0.4, 0.4, 0.4, 0.1,
                    )
                ], p=0.8),
                transforms_spatial.RandomGrayScale(p=0.2),
                transforms_spatial.RandomApply([
                    transforms_tensor.GaussianBlur((3, 3), (1.5, 1.5)).cuda()
                ], p=0.5),
                transforms_spatial.RandomHorizontalFlip(),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_transform(self, split='train'): #not here
        normalize = transforms_spatial.Normalize(
            self.cfg.get_list('dataset.mean'),
            self.cfg.get_list('dataset.std'),
            inplace=True,
        )

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')
        if split == 'train':
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoRandomCrop(
                    scale=(
                        st_cfg.get_float('crop_area.min'),
                        st_cfg.get_float('crop_area.max'),
                    )
                ),
            ])
        else:
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoCenterMaxCrop()
            ])

        if split == 'train': #not this
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomGrayScale(p=st_cfg.get_float('gray_scale')),
                transforms_spatial.ColorJitter(
                    brightness=st_cfg.get_float('color_jitter.brightness'),
                    contrast=st_cfg.get_float('color_jitter.contrast'),
                    saturation=st_cfg.get_float('color_jitter.saturation'),
                    hue=st_cfg.get_float('color_jitter.hue'),
                ),
                transforms_spatial.RandomHorizontalFlip(st_cfg.get_float('h_flip', default=0.5)),
                normalize,
            ])
        else:
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_temporal_transform(self, split): #not here
        tt_cfg = self.cfg.get_config('temporal_transforms')
        size = tt_cfg.get_int('size')
        tt_type = tt_cfg.get_string('type', default='clip')
        logger.info('Temporal transform type: %s', tt_type)

        if split == 'train':
            if tt_type == 'clip':
                crop = transforms_temporal.RandomStrideCrop(
                    size=size,
                    strides=tt_cfg.get_list('strides'),
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(size=size)
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        elif split in ['val', 'test']:
            if self.final_validate:
                n = tt_cfg.get_int('validate.final_n_crop')
            else:
                n = tt_cfg.get_int('validate.n_crop')

            if tt_type == 'clip':
                crop = transforms_temporal.EvenNCrop(
                    size=size,
                    stride=tt_cfg.get_int('validate.stride'),
                    n=n,
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(
                    size=size,
                    n_crop=n,
                )
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        else:
            raise ValueError(f'Unknown split "{split}"')
        return crop



class DataLoaderFactoryV6_visual:

    def __init__(self, cfg: ConfigTree, args, final_validate=False, debug=False):
        self.cfg = cfg
        self.final_validate = final_validate
        self.debug = debug
        self.args =args

    def build(self, vid=False, split='train', device=None):
        logger.info('Building Dataset: VID: %s, Split=%s', vid, split)

        ds_name = self.cfg.get_string('dataset.name')

        if vid: #here
            cpu_transform, gpu_transform = self.get_transform_vid()

            size = self.cfg.get_int('temporal_transforms.size')
            strides = self.cfg.get_list('temporal_transforms.strides')
            logger.info(f'Using frames: {size}, with strides: {strides}')
            temporal_transform = transforms_temporal.RandomStrideCrop(
                size=size,
                strides=strides,
            )
        else:
            cpu_transform, gpu_transform = self.get_transform(split)
            temporal_transform = self.get_temporal_transform(split)

        if ds_name == 'ucf101':
            from .ucf101 import UCF101
            ds = UCF101(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'vgg_sound':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_test':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split='test',
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_debug':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'trackingNet':
            from .trackingNet import TrackingNet
            ds = TrackingNet(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'davis':
            from .davis import davis
            ds = davis(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('kinetics'):  # [kinetics100, kinetics400]
            from .kinetics import Kinetics
            ds = Kinetics(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('hmdb51'):
            from .hmdb51 import HMDB51
            ds = HMDB51(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'smth_smth':
            from .smth_smth import Smth_Smth
            ds = Smth_Smth(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                split=split,
            )
        else:
            raise ValueError(f'Unknown dataset "{ds_name}"')
        #gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn(
        if int(self.args.davis)==1:
            gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn_mask2_fix_noaug(
                transform=gpu_transform,
                target_transform=(not vid),  # VID task does not need labels
                device=device,
            )
        else:
            gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn_mask2_fix(
                transform=gpu_transform,
                target_transform=(not vid),  # VID task does not need labels
                device=device,
            )
        #video_dataset = VideoDataset(
        if int(self.args.davis)==1:
            from .video_tracking_sal_light_davis import VideoDataset_constrained_mask_tracking_sal_light_davis
            video_dataset = VideoDataset_constrained_mask_tracking_sal_light_davis(
                samples=ds,
                temporal_transform=temporal_transform,
                spatial_transform=cpu_transform,
                num_clips_per_sample=2 if vid else 1,
                frame_rate=self.cfg.get_float('temporal_transforms.frame_rate'),
                args = self.args,
            )   
        else: 
            video_dataset = VideoDataset_constrained_mask_tracking_visual(
                samples=ds,
                temporal_transform=temporal_transform,
                spatial_transform=cpu_transform,
                num_clips_per_sample=2 if vid else 1,
                frame_rate=self.cfg.get_float('temporal_transforms.frame_rate'),
                args=self.args,
            ) 

        sampler = DistributedSampler(ds, shuffle=(split == 'train'))

        if split == 'train':
            batch_size = self.cfg.get_int('batch_size')
        elif self.final_validate:
            batch_size = self.cfg.get_int('final_validate.batch_size')
        else:
            batch_size = self.cfg.get_int('validate.batch_size')

        dl = DataLoader(
            video_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.get_int('num_workers'),
            sampler=sampler,
            drop_last=(split == 'train'),
            collate_fn=identity,  # We will deal with collation on main thread.
            multiprocessing_context=mp.get_context('fork'),
        )

        return MainProcessCollateWrapper(dl, gpu_collate_fn)


    def _get_normalize(self):
        if self.debug:
            def normalize(x):
                return x
        else:
            normalize = transforms_spatial.Normalize(
                self.cfg.get_list('dataset.mean'),
                self.cfg.get_list('dataset.std'),
                inplace=True,
            )
        return normalize

    def get_transform_vid(self): #here
        arch = self.cfg.get_string('model.arch')
        aug_plus = self.cfg.get_bool('moco.aug_plus')

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')

        logger.info(f'Training {arch} with size: {size}')

        cpu_transform = transforms_tensor.Compose([
            
            transforms_spatial.RawVideoRandomCrop(
                scale=(
                    0.4, 1.0
                )
            ),

        ])

        normalize = self._get_normalize()

        if not aug_plus: #this one
            gpu_transform = transforms_tensor.Compose([
                #transforms_spatial.ToTensorV2(),
                #transforms_spatial.Resize(size),
                transforms_spatial.RandomGrayScale(p=0.2),
                transforms_spatial.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.4,
                ),
                #transforms_spatial.RandomHorizontalFlip(),
                normalize,
            ])
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                #transforms_spatial.RandomGrayScale(p=0.2),
                #transforms_spatial.ColorJitter(
                #    brightness=0.4,
                #    contrast=0.4,
                #    saturation=0.4,
                #    hue=0.4,
                #),
                #transforms_spatial.RandomHorizontalFlip(),
                #normalize,
            ])
            """
            gpu_transform = transforms_tensor.Compose([
                #transforms_spatial.ToTensorV2(),
                #transforms_spatial.Resize(size),
                transforms_spatial.RandomHorizontalFlip(),
            ])
            """
            # for testing
            """
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                normalize,
            ])
            """
        else: 
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomApply([
                    transforms_spatial.ColorJitter(
                        0.4, 0.4, 0.4, 0.1,
                    )
                ], p=0.8),
                transforms_spatial.RandomGrayScale(p=0.2),
                transforms_spatial.RandomApply([
                    transforms_tensor.GaussianBlur((3, 3), (1.5, 1.5)).cuda()
                ], p=0.5),
                transforms_spatial.RandomHorizontalFlip(),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_transform(self, split='train'): #not here
        normalize = transforms_spatial.Normalize(
            self.cfg.get_list('dataset.mean'),
            self.cfg.get_list('dataset.std'),
            inplace=True,
        )

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')
        if split == 'train':
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoRandomCrop(
                    scale=(
                        st_cfg.get_float('crop_area.min'),
                        st_cfg.get_float('crop_area.max'),
                    )
                ),
            ])
        else:
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoCenterMaxCrop()
            ])

        if split == 'train': #not this
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomGrayScale(p=st_cfg.get_float('gray_scale')),
                transforms_spatial.ColorJitter(
                    brightness=st_cfg.get_float('color_jitter.brightness'),
                    contrast=st_cfg.get_float('color_jitter.contrast'),
                    saturation=st_cfg.get_float('color_jitter.saturation'),
                    hue=st_cfg.get_float('color_jitter.hue'),
                ),
                transforms_spatial.RandomHorizontalFlip(st_cfg.get_float('h_flip', default=0.5)),
                normalize,
            ])
        else:
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_temporal_transform(self, split): #not here
        tt_cfg = self.cfg.get_config('temporal_transforms')
        size = tt_cfg.get_int('size')
        tt_type = tt_cfg.get_string('type', default='clip')
        logger.info('Temporal transform type: %s', tt_type)

        if split == 'train':
            if tt_type == 'clip':
                crop = transforms_temporal.RandomStrideCrop(
                    size=size,
                    strides=tt_cfg.get_list('strides'),
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(size=size)
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        elif split in ['val', 'test']:
            if self.final_validate:
                n = tt_cfg.get_int('validate.final_n_crop')
            else:
                n = tt_cfg.get_int('validate.n_crop')

            if tt_type == 'clip':
                crop = transforms_temporal.EvenNCrop(
                    size=size,
                    stride=tt_cfg.get_int('validate.stride'),
                    n=n,
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(
                    size=size,
                    n_crop=n,
                )
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        else:
            raise ValueError(f'Unknown split "{split}"')
        return crop



class DataLoaderFactoryV3:

    def __init__(self, cfg: ConfigTree, args, final_validate=False, debug=False):
        self.cfg = cfg
        self.final_validate = final_validate
        self.debug = debug
        self.args = args

    def build(self, vid=False, split='train', device=None):
        logger.info('Building Dataset: VID: %s, Split=%s', vid, split)

        ds_name = self.cfg.get_string('dataset.name')

        if vid: #here
            cpu_transform, gpu_transform = self.get_transform_vid()

            size = self.cfg.get_int('temporal_transforms.size')
            strides = self.cfg.get_list('temporal_transforms.strides')
            logger.info(f'Using frames: {size}, with strides: {strides}')
            temporal_transform = transforms_temporal.RandomStrideCrop(
                size=size,
                strides=strides,
            )
        else:
            cpu_transform, gpu_transform = self.get_transform(split)
            temporal_transform = self.get_temporal_transform(split)

        if ds_name == 'ucf101':
            from .ucf101 import UCF101
            ds = UCF101(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'r2v2':
            from .r2v2 import R2V2
            ds = R2V2(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'mini_kinetics':
            from .mini_kinetics import mini_Kinetics
            ds = mini_Kinetics(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'mini_scene':
            from .mini_scene import mini_scene
            ds = mini_scene(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                split=split,
            )
        elif ds_name == 'davis':
            from .davis import davis
            ds = davis(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_test':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split='test',
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_debug':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'trackingNet':
            from .trackingNet import TrackingNet
            ds = TrackingNet(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_debug':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'kinetics_test':  # [kinetics100, kinetics400]
            from .kinetics import Kinetics_test
            ds = Kinetics_test(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('kinetics'):  # [kinetics100, kinetics400]
            from .kinetics import Kinetics
            ds = Kinetics(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('hmdb51'):
            from .hmdb51 import HMDB51
            ds = HMDB51(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name.startswith('jhmdb'):
            from .jhmdb import JHMDB
            ds = JHMDB(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'smth_smth':
            from .smth_smth import Smth_Smth
            ds = Smth_Smth(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                split=split,
            )
        else:
            raise ValueError(f'Unknown dataset "{ds_name}"')

        gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn(
            transform=gpu_transform,
            target_transform=(not vid),  # VID task does not need labels
            device=device,
        )
        video_dataset = VideoDataset(
        #video_dataset = VideoDataset_constrained(
            samples=ds,
            temporal_transform=temporal_transform,
            spatial_transform=cpu_transform,
            num_clips_per_sample=2 if vid else 1,
            frame_rate=self.cfg.get_float('temporal_transforms.frame_rate'),
            args=self.args
        )

        sampler = DistributedSampler(ds, shuffle=(split == 'train'))

        if split == 'train':
            batch_size = self.cfg.get_int('batch_size')
        elif self.final_validate:
            batch_size = self.cfg.get_int('final_validate.batch_size')
        else:
            batch_size = self.cfg.get_int('validate.batch_size')

        dl = DataLoader(
            video_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.get_int('num_workers'),
            sampler=sampler,
            drop_last=(split == 'train'),
            collate_fn=identity,  # We will deal with collation on main thread.
            multiprocessing_context=mp.get_context('fork'),
        )

        return MainProcessCollateWrapper(dl, gpu_collate_fn)


    def _get_normalize(self):
        if self.debug:
            def normalize(x):
                return x
        else:
            normalize = transforms_spatial.Normalize(
                self.cfg.get_list('dataset.mean'),
                self.cfg.get_list('dataset.std'),
                inplace=True,
            )
        return normalize

    def get_transform_vid(self): #here
        arch = self.cfg.get_string('model.arch')
        aug_plus = self.cfg.get_bool('moco.aug_plus')

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')

        logger.info(f'Training {arch} with size: {size}')

        cpu_transform = transforms_tensor.Compose([
            
            transforms_spatial.RawVideoRandomCrop(
                scale=(
                    0.4, 1.0
                )
            ),

        ])

        normalize = self._get_normalize()


        if not aug_plus: #this one
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomGrayScale(p=0.2),
                transforms_spatial.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.4,
                ),
                transforms_spatial.RandomHorizontalFlip(),
                normalize,
            ])
        else: 
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomApply([
                    transforms_spatial.ColorJitter(
                        0.4, 0.4, 0.4, 0.1,
                    )
                ], p=0.8),
                transforms_spatial.RandomGrayScale(p=0.2),
                transforms_spatial.RandomApply([
                    transforms_tensor.GaussianBlur((3, 3), (1.5, 1.5)).cuda()
                ], p=0.5),
                transforms_spatial.RandomHorizontalFlip(),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_transform(self, split='train'): #not here
        normalize = transforms_spatial.Normalize(
            self.cfg.get_list('dataset.mean'),
            self.cfg.get_list('dataset.std'),
            inplace=True,
        )

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')
        if split == 'train':
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoRandomCrop(
                    scale=(
                        st_cfg.get_float('crop_area.min'),
                        st_cfg.get_float('crop_area.max'),
                    )
                ),
            ])
        else:
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoCenterMaxCrop()
            ])

        if split == 'train': #mask should also follow this
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomGrayScale(p=st_cfg.get_float('gray_scale')),
                transforms_spatial.ColorJitter(
                    brightness=st_cfg.get_float('color_jitter.brightness'),
                    contrast=st_cfg.get_float('color_jitter.contrast'),
                    saturation=st_cfg.get_float('color_jitter.saturation'),
                    hue=st_cfg.get_float('color_jitter.hue'),
                ),
                transforms_spatial.RandomHorizontalFlip(st_cfg.get_float('h_flip', default=0.5)),
                normalize,
            ])
        else:
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_temporal_transform(self, split): #not here
        tt_cfg = self.cfg.get_config('temporal_transforms')
        size = tt_cfg.get_int('size')
        tt_type = tt_cfg.get_string('type', default='clip')
        logger.info('Temporal transform type: %s', tt_type)

        if split == 'train':
            if tt_type == 'clip':
                crop = transforms_temporal.RandomStrideCrop(
                    size=size,
                    strides=tt_cfg.get_list('strides'),
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(size=size)
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        elif split in ['val', 'test']:
            if self.final_validate:
                n = tt_cfg.get_int('validate.final_n_crop')
            else:
                n = tt_cfg.get_int('validate.n_crop')

            if tt_type == 'clip':
                crop = transforms_temporal.EvenNCrop(
                    size=size,
                    stride=tt_cfg.get_int('validate.stride'),
                    n=n,
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(
                    size=size,
                    n_crop=n,
                )
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        else:
            raise ValueError(f'Unknown split "{split}"')
        return crop


class DataLoaderFactoryV3_visual:

    def __init__(self, cfg: ConfigTree, args, final_validate=False, debug=False):
        self.cfg = cfg
        self.final_validate = final_validate
        self.debug = debug
        self.args = args

    def build(self, vid=False, split='train', device=None):
        logger.info('Building Dataset: VID: %s, Split=%s', vid, split)

        ds_name = self.cfg.get_string('dataset.name')

        if vid: #here
            cpu_transform, gpu_transform = self.get_transform_vid()

            size = self.cfg.get_int('temporal_transforms.size')
            strides = self.cfg.get_list('temporal_transforms.strides')
            logger.info(f'Using frames: {size}, with strides: {strides}')
            temporal_transform = transforms_temporal.RandomStrideCrop(
                size=size,
                strides=strides,
            )
        else:
            cpu_transform, gpu_transform = self.get_transform(split)
            temporal_transform = self.get_temporal_transform(split)

        if ds_name == 'ucf101':
            from .ucf101 import UCF101
            ds = UCF101(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'r2v2':
            from .r2v2 import R2V2
            ds = R2V2(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'mini_kinetics':
            from .mini_kinetics import mini_Kinetics
            ds = mini_Kinetics(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'mini_scene':
            from .mini_scene import mini_scene
            ds = mini_scene(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                split=split,
            )
        elif ds_name == 'davis':
            from .davis import davis
            ds = davis(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_test':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split='test',
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_debug':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'trackingNet':
            from .trackingNet import TrackingNet
            ds = TrackingNet(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'vgg_sound_debug':
            from .vgg_sound import VGG_Sound
            ds = VGG_Sound(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name == 'kinetics_test':  # [kinetics100, kinetics400]
            from .kinetics import Kinetics_test
            ds = Kinetics_test(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('kinetics'):  # [kinetics100, kinetics400]
            from .kinetics import Kinetics
            ds = Kinetics(
                self.cfg.get_string('dataset.root'),
                split=split,
                blacklist=self.cfg.get_list('dataset.blacklist'),
            )
        elif ds_name.startswith('hmdb51'):
            from .hmdb51 import HMDB51
            ds = HMDB51(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name.startswith('jhmdb'):
            from .jhmdb import JHMDB
            ds = JHMDB(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                fold=self.cfg.get_int('dataset.fold'),
                split=split,
            )
        elif ds_name == 'smth_smth':
            from .smth_smth import Smth_Smth
            ds = Smth_Smth(
                self.cfg.get_string('dataset.root'),
                self.cfg.get_string('dataset.annotation_path'),
                split=split,
            )
        else:
            raise ValueError(f'Unknown dataset "{ds_name}"')

        gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn_visual(
            transform=gpu_transform,
            target_transform=(not vid),  # VID task does not need labels
            device=device,
        )
        video_dataset = VideoDataset_visual(
        #video_dataset = VideoDataset_constrained(
            samples=ds,
            temporal_transform=temporal_transform,
            spatial_transform=cpu_transform,
            num_clips_per_sample=2 if vid else 1,
            frame_rate=self.cfg.get_float('temporal_transforms.frame_rate'),
            args=self.args
        )

        sampler = DistributedSampler(ds, shuffle=(split == 'train'))

        if split == 'train':
            batch_size = self.cfg.get_int('batch_size')
        elif self.final_validate:
            batch_size = self.cfg.get_int('final_validate.batch_size')
        else:
            batch_size = self.cfg.get_int('validate.batch_size')

        dl = DataLoader(
            video_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.get_int('num_workers'),
            sampler=sampler,
            drop_last=(split == 'train'),
            collate_fn=identity,  # We will deal with collation on main thread.
            multiprocessing_context=mp.get_context('fork'),
        )

        return MainProcessCollateWrapper(dl, gpu_collate_fn)


    def _get_normalize(self):
        if self.debug:
            def normalize(x):
                return x
        else:
            normalize = transforms_spatial.Normalize(
                self.cfg.get_list('dataset.mean'),
                self.cfg.get_list('dataset.std'),
                inplace=True,
            )
        return normalize

    def get_transform_vid(self): #here
        arch = self.cfg.get_string('model.arch')
        aug_plus = self.cfg.get_bool('moco.aug_plus')

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')

        logger.info(f'Training {arch} with size: {size}')

        cpu_transform = transforms_tensor.Compose([
            
            # transforms_spatial.RawVideoRandomCrop(
            #     scale=(
            #         0.4, 1.0
            #     )
            # ),

        ])

        normalize = self._get_normalize()

        # if not aug_plus: #this one
        #     gpu_transform = transforms_tensor.Compose([
        #         transforms_spatial.ToTensor(),
        #         transforms_spatial.Resize(size),
        #         transforms_spatial.RandomGrayScale(p=0.2),
        #         transforms_spatial.ColorJitter(
        #             brightness=0.4,
        #             contrast=0.4,
        #             saturation=0.4,
        #             hue=0.4,
        #         ),
        #         transforms_spatial.RandomHorizontalFlip(),
        #         normalize,
        #     ])
        if not aug_plus: #this one
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                # transforms_spatial.RandomGrayScale(p=0.2),
                # transforms_spatial.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4,
                #     hue=0.4,
                # ),
                # transforms_spatial.RandomHorizontalFlip(),
                #normalize,
            ])
        else: 
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                # transforms_spatial.RandomApply([
                #     transforms_spatial.ColorJitter(
                #         0.4, 0.4, 0.4, 0.1,
                #     )
                # ], p=0.8),
                # transforms_spatial.RandomGrayScale(p=0.2),
                # transforms_spatial.RandomApply([
                #     transforms_tensor.GaussianBlur((3, 3), (1.5, 1.5)).cuda()
                # ], p=0.5),
                # transforms_spatial.RandomHorizontalFlip(),
                # normalize,
            ])

        return cpu_transform, gpu_transform

    def get_transform(self, split='train'): #not here
        normalize = transforms_spatial.Normalize(
            self.cfg.get_list('dataset.mean'),
            self.cfg.get_list('dataset.std'),
            inplace=True,
        )

        st_cfg = self.cfg.get_config('spatial_transforms')
        size = st_cfg.get_int('size')
        if split == 'train':
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoRandomCrop(
                    scale=(
                        st_cfg.get_float('crop_area.min'),
                        st_cfg.get_float('crop_area.max'),
                    )
                ),
            ])
        else:
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoCenterMaxCrop()
            ])

        if split == 'train': #mask should also follow this
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                # transforms_spatial.RandomGrayScale(p=st_cfg.get_float('gray_scale')),
                # transforms_spatial.ColorJitter(
                #     brightness=st_cfg.get_float('color_jitter.brightness'),
                #     contrast=st_cfg.get_float('color_jitter.contrast'),
                #     saturation=st_cfg.get_float('color_jitter.saturation'),
                #     hue=st_cfg.get_float('color_jitter.hue'),
                # ),
                # transforms_spatial.RandomHorizontalFlip(st_cfg.get_float('h_flip', default=0.5)),
                # normalize,
            ])
        else:
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                #normalize,
            ])

        return cpu_transform, gpu_transform

    def get_temporal_transform(self, split): #not here
        tt_cfg = self.cfg.get_config('temporal_transforms')
        size = tt_cfg.get_int('size')
        tt_type = tt_cfg.get_string('type', default='clip')
        logger.info('Temporal transform type: %s', tt_type)

        if split == 'train':
            if tt_type == 'clip':
                crop = transforms_temporal.RandomStrideCrop(
                    size=size,
                    strides=tt_cfg.get_list('strides'),
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(size=size)
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        elif split in ['val', 'test']:
            if self.final_validate:
                n = tt_cfg.get_int('validate.final_n_crop')
            else:
                n = tt_cfg.get_int('validate.n_crop')

            if tt_type == 'clip':
                crop = transforms_temporal.EvenNCrop(
                    size=size,
                    stride=tt_cfg.get_int('validate.stride'),
                    n=n,
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(
                    size=size,
                    n_crop=n,
                )
            else:
                raise ValueError(f'Unknown temporal_transforms.type "{tt_type}"')
        else:
            raise ValueError(f'Unknown split "{split}"')
        return crop


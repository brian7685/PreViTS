import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
from pathlib import Path

import decord
import numpy as np
import torch
from datasets.transforms_video.transforms_temporal import Resample
from torch.utils.data import Dataset
from glob import glob
logger = logging.getLogger(__name__)
from PIL import Image
from . import functional as MF
from albumentations.augmentations import functional as AF
import cv2
from collections import Counter
decord.bridge.set_bridge('torch')
logger.info('Decord use torch bridge')
import json
import timeit
import random

@dataclass
class Sample:
    video_path: str
    class_index: int

def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img = cv2.resize(img, (width, height), interpolation=interpolation)
    return img


def bbox_from_mask(mask: np.ndarray):
    r"""
    Compute bounding box of the input mask, assumes mask is not all ``False``.
    Parameters
    ----------
    mask: np.ndarray
        Boolean mask of shape ``(height, width)`` with masked pixels ``True``.
    Returns
    -------
    Tuple[int, int, int, int]
        Absolute coordinates of bounding box, ``(x1, y1, x2, y2)``.
    """
    indices_height, indices_width = np.where(mask)
    top_left = (indices_width.min(), indices_height.min())
    bottom_right = (indices_width.max() + 1, indices_height.max() + 1)

    return (*top_left, *bottom_right)

class VideoDataset_constrained(Dataset):
    temporal_transform: Callable
    spatial_transform: Callable
    video_width: int = -1
    video_height: int = -1
    frame_rate: Optional[float] = None

    def __init__(
            self,
            samples: Sequence[Sample],
            temporal_transform=None,
            spatial_transform=None,
            video_width=-1,
            video_height=-1,
            num_clips_per_sample=1,
            frame_rate=None
    ):
        """
        For VID task: num_clips_per_sample = 2
        For finetune 10 crops validation，num_clips_per_sample = 1，but "temporal_transform" will output 10 times longer video
        """
        self.samples = samples
        self.num_clips_per_sample = num_clips_per_sample
        self.video_width = video_width
        self.video_height = video_height
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_rate = frame_rate
        self.resample_fps = Resample(target_fps=frame_rate)
        #load max mask here
        #with open('/export/home/RSPNet/meta_data/max_obj.json') as f:
        with open('/export/home/RSPNet/meta_data/max_obj_train_split.json') as f:
            max_obj = json.load(f)
        self.max_obj = max_obj
        #with open('/export/home/ExtremeNet/vgg_frame_count_test_split.json') as f:
        with open('/export/home/ExtremeNet/vgg_frame_count_train_split.json') as f:
            frame_count = json.load(f)
            
        self.frame_count = frame_count
       

        logger.info(f'You are using VideoDataset: {self.__class__}')

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        sample: Sample = self.samples[index]
        vr = decord.VideoReader(
            str(sample.video_path),
            width=self.video_width,
            height=self.video_height,
            num_threads=1,
        )
        num_frames = len(vr)
        if num_frames == 0:
            raise Exception(f'Empty video: {sample.video_path}')
        frame_indices = np.arange(num_frames)  # [0, 1, 2, ..., N - 1]

        if self.frame_rate is not None: #null, not in
            frame_indices = self.resample_fps(frame_indices, vr.get_avg_fps())
        #sample two segments
        """
        clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
    
        clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
        clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel

        clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]

        clip_list = [self.spatial_transform(clip) for clip in clip_list]

        return clip_list, sample.class_index
        """
        load_mask=1 

        #print('sample.video_path',sample.video_path)
        v_path = str(sample.video_path)
        vid = v_path.split('/')[-1].split('.')[0]
        #print('vid',vid)
        mid_n = 16
        frame_length = 32
        zero_mask_one = np.zeros((1,32,224,224), dtype=np.uint8)
        zero_mask = [zero_mask_one,zero_mask_one]
        class_name = v_path.split('/')[-2]


        if class_name+'|'+vid not in self.max_obj.keys():
            #print('no seg',class_name,vid)
            clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
            clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
            clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
            clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
            clip_list = [self.spatial_transform(clip) for clip in clip_list]
            
            return clip_list, zero_mask,0,0,sample.class_index

        max_key = self.max_obj[class_name+'|'+vid]#[vid]
        frame_count = self.frame_count[class_name+'|'+vid]
        start = timeit.default_timer()


        #print(class_name,vid)
        if 1:#vid=='01SaXyGY5SM_000030':#'D2ISzNuhtxE_000009':
            #print('class',class_name,vid)
            mask_l = {}
            for name in glob('/export/home/b_data/vgg_sound-256/seg/test_split/'+class_name+'/'+vid+'/*'):
                #print(name)
                frame_n = int(name.split('_')[-1])
                mask_l[frame_n]=name
            #print('mask_path',name)
            #mask_l = sorted(mask_l)
            stop0 = timeit.default_timer()
            #print('load mask list Time: ', stop0 - start) 
            """
            object_id = Counter()
            for name in glob('/export/home/b_data/vgg_sound-256/seg/test_split/'+class_name+'/'+vid+'/*/*'):
                class_id=name.split('/')[-1].split('_')[2]
                inst_id = name.split('/')[-1].split('_')[3]
                CID = class_id+'_'+inst_id
                object_id[CID]+=1

            
            try:
                max_key = max(object_id, key=object_id.get) #get key with max value.
            except:
                print('c_name',class_name,vid,object_id)
                clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
    
                clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
                clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel

                clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]

                clip_list = [self.spatial_transform(clip) for clip in clip_list]

                return clip_list, sample.class_index
            #print('max_key',max_key)
            """
        if len(mask_l)/float(frame_count)<0.5: #can be replaced by json metadata
            #print('few seg',class_name,vid,len(mask_l))
            clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
            clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
            clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
            clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
            clip_list = [self.spatial_transform(clip) for clip in clip_list]
            return clip_list, zero_mask,0,0,sample.class_index
        #print(class_name,vid,max_key)
        stop = timeit.default_timer()
        #print('Find max object Time: ', stop - stop0)  
        #print('frame_indices',frame_indices)
        #original 16, 16*speed(2) = 32 length
        clip_frame_indices_list = []
        obj_set = set()
        #Your statements here

        
        max_key_c = max_key.split('_')[0]
        if 1:#vid=='01SaXyGY5SM_000030':
            match=0
            too_much = 0
            while match==0:
                too_much+=1
                if too_much>100:
                    print('too much',class_name,vid,mid_frame_q,max_key)
                    clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
                    clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
                    clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
                    clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
                    clip_list = [self.spatial_transform(clip) for clip in clip_list]
                    return clip_list, zero_mask,0,0,sample.class_index
                t_crop = self.temporal_transform(frame_indices)
                mid_frame_q = t_crop[mid_n]
                #print('mid_frame_q',mid_frame_q)
                #print('vid',vid)
                if mid_frame_q not in mask_l.keys():
                    continue
                
                mask_loc_q = mask_l[mid_frame_q]

                mask_img = mask_loc_q.split('/')[-1]
                name = mask_loc_q+'/'+mask_img+'.jpg_'+max_key+'_mask.png'
                try:
                    gt_mask = np.array(Image.open(name).convert('L'))
                except:
                    continue
                mask_area = np.count_nonzero(gt_mask)
                if mask_area==0:
                    continue

                
                for name in glob(mask_loc_q+'/*'):
                    #has same instance

                    obj_c = name.split('/')[-1].split('_')[2]
                    #print('max_key',vid,name,max_key,obj_c)
                    #obj_set.add(obj_c)
                    if obj_c ==max_key_c:#in obj_set:
                        clip_frame_indices_list.append(t_crop) 
                        #print('match')
                        match=1
                        break
                
            #print(class_name,vid,max_key,'match1')
            #clip_frame_indices_list.append(t_crop) 
            #for _ in range(self.num_clips_per_sample):
            match=0
            too_much = 0
            while match==0:
                too_much+=1
                if too_much>100:
                    print('too much',class_name,vid,mid_frame_q,max_key)
                    clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
                    clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
                    clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
                    clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
                    clip_list = [self.spatial_transform(clip) for clip in clip_list]
                    return clip_list, zero_mask,0,0,sample.class_index

                    
                t_crop = self.temporal_transform(frame_indices)
                mid_frame_k = t_crop[mid_n]
                #print('mid_frame_k',vid,len(mask_l),mid_frame_k)
                if mid_frame_k not in mask_l.keys():
                    continue


                mask_loc_k = mask_l[mid_frame_k]
                mask_img = mask_loc_k.split('/')[-1]
                name = mask_loc_k+'/'+mask_img+'.jpg_'+max_key+'_mask.png'

                #gt_mask = np.array(Image.open(name).convert('L'))
                try:
                    gt_mask = np.array(Image.open(name).convert('L'))
                except:
                    continue
                mask_area = np.count_nonzero(gt_mask)
                if mask_area==0:
                    continue
                
                
                for name in glob(mask_loc_k+'/*'):
                    #has same instance
                    obj_c = name.split('/')[-1].split('_')[2]
                    if obj_c ==max_key_c:#in obj_set:
                        clip_frame_indices_list.append(t_crop) 
                        #print('match')
                        match=1
                        break
                
            #print(class_name,vid,max_key,'match2')
        #else:
        #    clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
        stop2 = timeit.default_timer()
        #print('Temporal sampling Time: ', stop2 - stop)  
        #iterate this step and pick the highest IOU
         
        #array([176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, array([ 94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
        #if vid=='D2ISzNuhtxE_000009':
        #    print('clip_frame_indices_list',clip_frame_indices_list) 
        #    print('num_frames',num_frames)
            #print('mask_l',mask_l)
        # Fetch all frames in one `vr.get_batch` call
        clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
        clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
        #print(clips.shape) #([64, 256, 456, 3])
        clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
        clip_list_temp = clip_list
        #clip_list = [self.spatial_transform(clip) for clip in clip_list]
        if 1:#vid=='01SaXyGY5SM_000030':
            clip_list = []
            #for clip in clip_list:
            mask_locs = [mask_loc_q,mask_loc_k]
            gt_masks = []
            mask_list = []
            for k in range(2):
                IOU_max = 0
                #for l in range(10):
                #crop_T,top, left, h, w = self.spatial_transform(clip_list_temp[k])
                #i, j, h, w = crop_T.get_params
                #clip_list.append(crop_T)
                #"""
                #print('mask_locs[k]',mask_locs[k])
                #print('crop',top, left, h, w)
                mask_img = mask_locs[k].split('/')[-1]
                name = mask_locs[k]+'/'+mask_img+'.jpg_'+max_key+'_mask.png'
                #print(name)
                mask_list.append(name)
                gt_mask = np.array(Image.open(name).convert('L'))
                gt_masks.append(gt_mask)
                # binarize mask
                gt_mask[gt_mask <= 128] = 0      # Black
                gt_mask[gt_mask > 128] = 1     # White
                #print('gt_mask',gt_mask.shape)

                if k==0:
                    try:
                        first_crop_coords = MF.get_random_crop_coords_mask_cover(
                            gt_mask,
                            gt_mask,
                            other_coords=None,
                            min_areacover=0.2,
                            scale=(0.4, 1.0),
                            ratio=(3. / 4., 4. / 3.),
                        )
                        #print('first_crop_coords',first_crop_coords)
                        left,top,right,bot = first_crop_coords
                        query_gt_mask=gt_mask
                    except:
                        print('crop error',class_name,vid,max_key,name)
                        clip_list = [self.spatial_transform(clip) for clip in clip_list_temp]
                        return clip_list, zero_mask,0,0,sample.class_index
                if k==1:
                    try:
                        second_crop_coords = MF.get_random_crop_coords_mask_cover(
                        gt_mask,
                        gt_mask,
                        other_coords=None,
                        min_areacover=0.2,
                        scale=(0.4, 1.0),
                        ratio=(3. / 4., 4. / 3.),
                        )
                        #print('second_crop_coords',second_crop_coords)
                        left,top,right,bot = second_crop_coords
                        key_gt_mask = gt_mask
                    except:
                        print('crop error',class_name,vid,max_key,name)
                        clip_list = [self.spatial_transform(clip) for clip in clip_list_temp]
                        return clip_list, zero_mask,0,0,sample.class_index
                region = clip_list_temp[k][..., top: bot, left: right, :]
                crop_T_max = region.contiguous()
                clip_list.append(crop_T_max)
            #print('t1, t2, ltrb1, ltrb2', mid_frame_q, mid_frame_k, first_crop_coords, second_crop_coords)
            Path('crop_json/'+class_name).mkdir(parents=True, exist_ok=True)
            left1,top1,right1,bot1 = first_crop_coords
            left2,top2,right2,bot2 = second_crop_coords
            data ={'vid':vid,
                't1':int(mid_frame_q),\
            't2':int(mid_frame_k),\
            'left1':int(left1),'top1':int(top1),'right1':int(right1),'bot1':int(bot1),\
            'left2':int(left2),'top2':int(top2),'right2':int(right2),'bot2':int(bot2),\
            'mask1':mask_list[0],'mask2':mask_list[1]
            }
            """
            with open('crop_json/'+class_name+'/'+vid+'.json', 'w') as outfile:
                
                json.dump(data, outfile)
            #"""
            """
            bot = top+h
            right = left+w
            crop_region = np.zeros((gt_mask.shape[0],gt_mask.shape[1]))
            for i in range(gt_mask.shape[0]):
                for j in range(gt_mask.shape[1]):
                    if i>=top and i < bot and j > left and j <right:
                        crop_region[i][j]=1
            

            intersection = (crop_region * gt_mask).sum()
            total = (crop_region + gt_mask).sum()
            union = total - intersection 
            
            IoU = (intersection )/(union )
            print('IOU',IoU)
            if IoU>IOU_max:
                crop_T_max = crop_T
                IOU_max=IoU
            """

            
                #"""
        #else:
        #    clip_list = [self.spatial_transform(clip) for clip in clip_list]
        stop3 = timeit.default_timer()
        #print('Spatial sampling Time: ', stop3 - stop2) 
        # ====================== MASK ========================================
        #try:
        if load_mask==1:
            
            query_mask_list = []
            #query_frame_list = []
            frame_path_list = []
            if 1:
                mask_frame = data['mask1'].split('/')[-2]
                mask_frame_n = int(mask_frame.split('_')[-1])
                query_mask_path = data['mask1']
                new_path = '/export/home/RSPNet/meta_data/vgg_meta/test_split/'+class_name+'/'+vid+'.npy'
                try:
                    np_mask = np.load(new_path)['arr_0']
                    #print('np_mask',np_mask.shape)
                except:
                    #print('no mask',new_path)
                    info = [class_name,vid,0,0,data]

                    return clip_list, zero_mask, 0, info,sample.class_index
                stop3 = timeit.default_timer()
                query_mask_array = np.zeros((frame_length, 224, 224))
                for i in range(frame_length):
                    cur_num = str(mask_frame_n-mid_n+i+1).zfill(5)
                    cur_num_i = int(cur_num)-1
                    frame_name = 'image_'+cur_num

                    new_path = query_mask_path.replace(mask_frame,frame_name)
                    max_id = new_path.split('jpg')[-1].split('mask')[0][1:-1]
                    all_mask_name = new_path.replace(max_id,'*')

                    frame_path = '/export/home/data/vgg_sound-256/frame/test_split/'+class_name+\
                    '/'+data['vid']+'/'+frame_name+'.jpg'

                    frame_path_list.append(frame_path)

                    
                    query_mask = np_mask[cur_num_i,data['top1']:data['bot1'],data['left1']:data['right1']]
                    query_mask = resize(query_mask, 224, 224)
                    query_mask[query_mask <= 128] = 0      # Black
                    query_mask[query_mask > 128] = 1
                    query_mask_array[i,:,:] = query_mask
                    
            if 1:
            #try:
                mask_frame = data['mask2'].split('/')[-2]
                mask_frame_n = int(mask_frame.split('_')[-1])
                #print('key_mask_frame',mask_frame)
                key_mask_path = data['mask2']
                key_mask_array = np.zeros((frame_length, 224, 224))
                for i in range(frame_length):
                    cur_num = str(mask_frame_n-mid_n+i+1).zfill(5)
                    frame_name = 'image_'+cur_num
                    cur_num_i = int(cur_num)-1
                    #print(frame_name)
                    #print('query_mask_path',query_mask_path)
                    #print('mask_frame',mask_frame)
                    new_path = key_mask_path.replace(mask_frame,frame_name)
                    max_id = new_path.split('jpg')[-1].split('mask')[0][1:-1]
                    all_mask_name = new_path.replace(max_id,'*')


                    key_mask = np_mask[cur_num_i,data['top2']:data['bot2'],data['left2']:data['right2']]
                    key_mask = resize(key_mask, 224, 224)
                    key_mask[key_mask <= 128] = 0      # Black
                    key_mask[key_mask > 128] = 1
                    key_mask_array[i,:,:] = key_mask

                query_mask_array = np.expand_dims(query_mask_array, axis=0).astype(np.uint8)
                key_mask_array = np.expand_dims(key_mask_array, axis=0).astype(np.uint8)
                mask_list = [query_mask_array,key_mask_array]
                #print('query_mask_array',query_mask_array.dtype)
                #print('mask_list',mask_list)
                #if index==21:
                #    print('index',index)
                #    print('clip_frame_indices_list',clip_frame_indices_list)
                #print(class_name,vid,'finish')
                info = [class_name,vid,data['mask1'],frame_path_list,data]
                
                stop4 = timeit.default_timer()
                #print('mask looading Time: ', stop4 - stop3) 
                return clip_list, mask_list, load_mask, info,sample.class_index #clip_list = two video segment
        else:
            info = [class_name,vid,0,0,data]
            #print('zero_mask',zero_mask[0].dtype)
            return clip_list, zero_mask, 0, info,sample.class_index

    def __len__(self):
        return len(self.samples)

def rand_temp(frame_indices):
    num_frames = len(num_frames)
    needed_frames = 32
    start_index = random.randint(0, num_frames - needed_frames) #300-32
    selected = np.arange(start_index, start_index + needed_frames, stride)
    #print('frame_indices[selected]',frame_indices[selected])
    return frame_indices[selected]

class VideoDataset(Dataset):
    temporal_transform: Callable
    spatial_transform: Callable
    video_width: int = -1
    video_height: int = -1
    frame_rate: Optional[float] = None

    def __init__(
            self,
            samples: Sequence[Sample],
            temporal_transform=None,
            spatial_transform=None,
            video_width=-1,
            video_height=-1,
            num_clips_per_sample=1,
            frame_rate=None,
            args=None
    ):
        """
        For VID task: num_clips_per_sample = 2
        For finetune 10 crops validation，num_clips_per_sample = 1，but "temporal_transform" will output 10 times longer video
        """
        self.samples = samples
        self.num_clips_per_sample = num_clips_per_sample
        self.video_width = video_width
        self.video_height = video_height
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_rate = frame_rate
        self.resample_fps = Resample(target_fps=frame_rate)
        self.args = args

       

        logger.info(f'You are using VideoDataset: {self.__class__}')

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        sample: Sample = self.samples[index]
        try:
            vr = decord.VideoReader(
                str(sample.video_path),
                width=self.video_width,
                height=self.video_height,
                num_threads=1,
            )
        except:
            print('error',str(sample.video_path))
        #print('sample.video_path',sample.video_path)
        v_path = str(sample.video_path)
        vid = v_path.split('/')[-1].split('.')[0]
        #print('vid',vid)

        class_name = v_path.split('/')[-2]
        #print('class_name',class_name)
        # if vid=='D2ISzNuhtxE_000009':
        #     #print('index21',class_name,vid)
        #     mask_l = []
        #     for name in glob('/export/home/b_data/vgg_sound-256/seg/test_split/'+class_name+'/'+vid+'/*'):
        #         #print(name)
        #         mask_l.append(name)
        #     mask_l = sorted(mask_l)


        num_frames = len(vr)
        if num_frames == 0:
            raise Exception(f'Empty video: {sample.video_path}')
        frame_indices = np.arange(num_frames)  # [0, 1, 2, ..., N - 1]
        
        if int(self.args.rand_f)==1:
            #frame_indices_t = frame_indices
            np.random.shuffle(frame_indices)
            #frame_indices = np.random.randint(num_frames, size=num_frames)
            #print('frame_indices_t',frame_indices_t,frame_indices)

        if self.frame_rate is not None: #null, not in
            frame_indices = self.resample_fps(frame_indices, vr.get_avg_fps())
        #sample two segments
        
        #print('frame_indices',frame_indices)
        #original 16, 16*speed(2) = 32 length

        #fixed temporal transform
        if int(self.args.fixed_temp)==1:
            #print('fix temp')
            
            v_temp = self.temporal_transform(frame_indices)
            clip_frame_indices_list = [v_temp for _ in range(self.num_clips_per_sample)]
        
        else:
            clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
        
        #iterate this step and pick the highest IOU
         
        #array([176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, array([ 94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,

        # Fetch all frames in one `vr.get_batch` call
        clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
        clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
        #print(clips.shape) #([64, 256, 456, 3])
        clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]

        clip_list = [self.spatial_transform(clip) for clip in clip_list]
        #if index==21:
        #    print('index',index)
        #    print('clip_frame_indices_list',clip_frame_indices_list)
        return clip_list, sample.class_index

    def __len__(self):
        return len(self.samples)

class VideoDataset_visual(Dataset):
    temporal_transform: Callable
    spatial_transform: Callable
    video_width: int = -1
    video_height: int = -1
    frame_rate: Optional[float] = None

    def __init__(
            self,
            samples: Sequence[Sample],
            temporal_transform=None,
            spatial_transform=None,
            video_width=-1,
            video_height=-1,
            num_clips_per_sample=1,
            frame_rate=None,
            args=None
    ):
        """
        For VID task: num_clips_per_sample = 2
        For finetune 10 crops validation，num_clips_per_sample = 1，but "temporal_transform" will output 10 times longer video
        """
        self.samples = samples
        self.num_clips_per_sample = num_clips_per_sample
        self.video_width = video_width
        self.video_height = video_height
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_rate = frame_rate
        self.resample_fps = Resample(target_fps=frame_rate)
        self.args = args

       

        logger.info(f'You are using VideoDataset: {self.__class__}')

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        sample: Sample = self.samples[index]
        try:
            vr = decord.VideoReader(
                str(sample.video_path),
                width=self.video_width,
                height=self.video_height,
                num_threads=1,
            )
        except:
            print('error',str(sample.video_path))
        #print('sample.video_path',sample.video_path)
        v_path = str(sample.video_path)
        vid = v_path.split('/')[-1].split('.')[0]
        #print('vid',vid)

        class_name = v_path.split('/')[-2]

        num_frames = len(vr)
        if num_frames == 0:
            raise Exception(f'Empty video: {sample.video_path}')
        if num_frames>64:
            frame_indices = np.arange(64)
        else:
            frame_indices = np.arange(num_frames)  # [0, 1, 2, ..., N - 1]
        
        if int(self.args.rand_f)==1:
            #frame_indices_t = frame_indices
            np.random.shuffle(frame_indices)
            #frame_indices = np.random.randint(num_frames, size=num_frames)
            #print('frame_indices_t',frame_indices_t,frame_indices)

        if self.frame_rate is not None: #null, not in
            frame_indices = self.resample_fps(frame_indices, vr.get_avg_fps())
        #sample two segments
        
        #print('frame_indices',frame_indices)
        #original 16, 16*speed(2) = 32 length

        #fixed temporal transform
        if int(self.args.fixed_temp)==1:
            #print('fix temp')
            
            v_temp = self.temporal_transform(frame_indices)
            clip_frame_indices_list = [v_temp for _ in range(self.num_clips_per_sample)]
        
        else:
            clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
        
        #iterate this step and pick the highest IOU
         
        #array([176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, array([ 94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
        #clip_frame_indices_list = [frame_indices[:64]]#[for i in range()]
        # Fetch all frames in one `vr.get_batch` call
        clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
        clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
        #print(clips.shape) #([64, 256, 456, 3])
        clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
        clip_list_temp = clip_list
        region = clip_list_temp[0]#[..., top: bot, left: right, :]
        crop_T_max = region.contiguous()
        clip_list = []
        clip_list.append(crop_T_max)
        #clip_list = [self.spatial_transform(clip) for clip in clip_list]
        #if index==21:
        #    print('index',index)
        #    print('clip_frame_indices_list',clip_frame_indices_list)
        #print('vid',vid)
        return clip_list, sample.class_index,class_name, vid,clip_frame_indices_list

    def __len__(self):
        return len(self.samples)
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
import os

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

def temp_trans(frame_indices,frame_length, withSeg):
    num_frames = len(frame_indices)
    needed_frames = frame_length
    rand = random.random()
    mid = 0
    while mid+16 >= num_frames or mid-16 <0:
        mid = random.choice(withSeg)-1 #to index	
    #start_index = random.randint(0, num_frames - needed_frames) #300-32
    #selected = np.arange(start_index, start_index + needed_frames, 1)
    selected = np.arange(mid-16, mid+16, 1)
    #print('frame_indices[selected]',frame_indices[selected])
    return frame_indices[selected]

class VideoDataset_constrained_mask_tracking_sal_light_davis(Dataset):
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
        self.args = args
        self.samples = samples
        self.num_clips_per_sample = num_clips_per_sample
        self.video_width = video_width
        self.video_height = video_height
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_rate = frame_rate
        self.resample_fps = Resample(target_fps=frame_rate)
        #load max mask here
        if args.multi==1:
            with open('/export/home/RSPNet/meta_data/freq_obj_test_split.json') as f:
                freq_obj = json.load(f)
            self.freq_obj = freq_obj
        else:
            with open('/export/home/RSPNet/meta_data/max_obj.json') as f:
                max_obj = json.load(f)
            self.max_obj = max_obj
        # with open('/export/home/ExtremeNet/vgg_frame_count_test_split.json') as f:
        #     frame_count = json.load(f)
        # self.frame_count = frame_count
        #with open('/export/home/RSPNet/meta_data_op/withSeg_test_split_sal_0819.json') as f:

        if 'kinetics' in str(self.args.experiment_dir):
            if 'train' in str(self.args.experiment_dir):
                with open('/export/home/RSPNet/meta_data_op/withSeg_train_split_sal_track_k.json') as f:
                    withSeg = json.load(f)
            else:
                with open('/export/home/RSPNet/meta_data_op/withSeg_test_split_sal_track_k.json') as f:
                    withSeg = json.load(f)
        elif 'r2v2' in str(self.args.experiment_dir):
            with open('/export/home/RSPNet/meta_data_op/withSeg_train_split_sal_track_r.json') as f:
                    withSeg = json.load(f)
        else:
            if 'supervised' in str(self.args.experiment_dir):
                with open('/export/home/RSPNet/meta_data_op/withSeg_train_split_0813.json') as f:
                    withSeg = json.load(f)
            elif 'strict' in str(self.args.experiment_dir):
                with open('/export/home/RSPNet/meta_data_op/withSeg_train_split_sal_track_strict.json') as f:
                    withSeg = json.load(f)
            elif 'train' in str(self.args.experiment_dir):
                with open('/export/home/RSPNet/meta_data_op/withSeg_train_split_sal_track.json') as f:
                    withSeg = json.load(f)
            else:
                with open('/export/home/RSPNet/meta_data_op/withSeg_test_split_sal_track.json') as f:
                    withSeg = json.load(f)
        self.withSeg = withSeg


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
        if int(self.args.onlycc)==1:
            load_mask=0 
        else:
            load_mask=1

        if int(self.args.nocc)==1:
            nocc=1 
        else:
            nocc=0
        #load_mask=0
        #print('sample.video_path',sample.video_path)
        v_path = str(sample.video_path)
        vid = v_path.split('/')[-1].split('.')[0]
        #print('vid',vid)
        mid_n = int(self.args.mid_n)#16
        frame_length = 1
        zero_mask_one = np.zeros((1,1,224,224), dtype=np.uint8)
        zero_mask = [zero_mask_one,zero_mask_one]
        class_name = v_path.split('/')[-2]
        ver = int (self.args.ver)
        # test baseline, need to remove this later
        """
        clip_frame_indices_list = [self.temporal_transform(frame_indices) for _ in range(self.num_clips_per_sample)]
        clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
        clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
        clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
        clip_list = [self.spatial_transform(clip) for clip in clip_list]
        
        return clip_list, zero_mask,0,0,sample.class_index
        """
        rand = random.random()
        if rand>0.5:
            self.args.split=1
        else:
            self.args.split=0
        
        #print('yes seg',class_name,vid)
        #max_key = self.max_obj[class_name+'|'+vid]#[vid]
        #frame_count = self.frame_count[class_name+'|'+vid]
        start = timeit.default_timer()
        #withSeg = self.withSeg[class_name+'|'+vid]
        
        #withSeg_thre = int(self.args.withSeg_thre)


        #print(class_name,vid,max_key)
        #print('wrong')
        stop = timeit.default_timer()
        #print('Find max object Time: ', stop - stop0)  
        #print('frame_indices',frame_indices)
        #original 16, 16*speed(2) = 32 length
        clip_frame_indices_list = []
        obj_set = set()
        #Your statements here

        
        #max_key_c = max_key.split('_')[0]
        gt_masks = []
        mask_list= []
        area_list = []
        if 1:#vid=='01SaXyGY5SM_000030':
            match=0
            too_much = 0
            while match==0:
                too_much+=1
                
                t_crop = self.temporal_transform(frame_indices)
                frame_no = int(self.args.frame_no)
                if frame_no!=-1:
                    if not frame_no>=len(frame_indices)-1:
                        t_crop = [int(self.args.frame_no)]#[35]
                t_crop_q = t_crop
                #t_crop = temp_trans(frame_indices,frame_length, withSeg)
                mid_frame_q = t_crop[mid_n] 
                
                cur_num = str(mid_frame_q).zfill(5)
                if '17' in str(self.args.experiment_dir):
                    if int(self.args.split)==1:
                        name = '/export/home/davis/split/'+vid+'/'+cur_num+'.png/'+str(self.args.split)+'.png'
                        if not os.path.isfile(name):
                            name = '/export/home/davis/split/'+vid+'/'+cur_num+'.png/0.png'
                    else:
                        name = '/export/home/davis/split/'+vid+'/'+cur_num+'.png/0.png'
                else:
                    name = '/export/home/motiongrouping/DAVIS/Annotations/'+class_name+'/'+vid+'/'+cur_num+'.png'
                try:
                    #print('open query tracking')
                    gt_mask = np.array(Image.open(name).convert('L')).astype(np.uint8)
                    
                except:
                    print('query error', name)
                    continue
                #gt_mask_b = gt_mask
                # gt_mask[gt_mask <= 128] = 0      # Black
                # gt_mask[gt_mask > 128] = 1     # White
                mask_area = np.count_nonzero(gt_mask)
                if mask_area<10:
                   continue
                else:
                    mask_list.append(name)
                    gt_masks.append(gt_mask)
                    match=1
                    clip_frame_indices_list.append(t_crop) 
                    area_list.append(mask_area)

            match=0
            too_much = 0
            while match==0:
                too_much+=1
                

                if int(self.args.cast)==1:
                    t_crop = t_crop_q
                elif int(self.args.constant)==1:
                    mid_temp = t_crop_q[mid_n] + 1
                    if mid_temp + 50< len(frame_indices):
                        t_crop = t_crop_q+50
                    else:
                        t_crop = t_crop_q-50
                else:
                    t_crop = self.temporal_transform(frame_indices)
                
                t_crop = [0]#[len(frame_indices)-1]#[int(frame_indices[int(len(frame_indices)/2)])]#[int(len(frame_indices)/2)]##len()#[0]
                t_crop_k = t_crop
                #t_crop = temp_trans(frame_indices,frame_length, withSeg)
                mid_frame_k = t_crop[mid_n] 
                #print('mid_frame_k',vid,len(mask_l),mid_frame_k)
                #if mid_frame_k not in mask_l.keys():
                #    continue

                if int(self.args.time)==1:
                    if abs(mid_frame_k-mid_frame_q)>int(self.args.interval):
                        continue
                

                #mask_loc_k = mask_l[mid_frame_k]
                #mask_img = mask_loc_k.split('/')[-1]
                #name = mask_loc_k+'/'+mask_img+'.jpg_'+max_key+'_mask.png'
                #cur_num = 'image_'+str(mid_frame_k).zfill(5)
                cur_num =str(mid_frame_k).zfill(5)
                #sftp://localhost:2222/export/optical-flow/RSPNet/meta_data/vgg_sal_test_track/raining/A-QYRlK7cd8_000030/00019.png
                #name = '/export/optical-flow/data/vgg_sal_test_track/'+class_name+'/'+vid+'/'+cur_num+'.png'
                if '17' in str(self.args.experiment_dir):
                    if int(self.args.split)==1:
                        name = '/export/home/davis/split/'+vid+'/'+cur_num+'.png/'+str(self.args.split)+'.png'
                        if not os.path.isfile(name):
                            name = '/export/home/davis/split/'+vid+'/'+cur_num+'.png/0.png'
                    else:
                        name = '/export/home/davis/split/'+vid+'/'+cur_num+'.png/0.png'
                else:
                    name = '/export/home/motiongrouping/DAVIS/Annotations/'+class_name+'/'+vid+'/'+cur_num+'.png'
                try:
                    #print('open tracking')
                    gt_mask = np.array(Image.open(name).convert('L')).astype(np.uint8)
                    
                except:
                    print('key error', name )
                    continue
                # gt_mask[gt_mask <= 128] = 0      # Black
                # gt_mask[gt_mask > 128] = 1     # White
                
                mask_area = np.count_nonzero(gt_mask)
                if mask_area<10:
                   continue
                else:
                    mask_list.append(name)
                    gt_masks.append(gt_mask)
                    match=1
                    clip_frame_indices_list.append(t_crop) 
                    area_list.append(mask_area)
                

        clip_frame_indices = np.concatenate(clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
        clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C] N video * T time, Height, Width, Channel
        #print(clips.shape) #([64, 256, 456, 3])
        clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]
        clip_list_temp = clip_list
        #clip_list = [self.spatial_transform(clip) for clip in clip_list]
        if 1:#vid=='01SaXyGY5SM_000030':
            clip_list = []
            #for clip in clip_list:
            #mask_locs = [mask_loc_q,mask_loc_k]
            #gt_masks = []
            #mask_list = []
            for k in range(2):
                IOU_max = 0
                #for l in range(10):
                #crop_T,top, left, h, w = self.spatial_transform(clip_list_temp[k])
                #i, j, h, w = crop_T.get_params
                #clip_list.append(crop_T)
                #"""
                #print('mask_locs[k]',mask_locs[k])
                #print('crop',top, left, h, w)
                #mask_img = mask_locs[k].split('/')[-1]
                #name = mask_locs[k]+'/'+mask_img+'.jpg_'+max_key+'_mask.png'
                #print(name)
                
                gt_mask = gt_masks[k]#np.array(Image.open(name).convert('L'))
                #gt_masks.append(gt_mask)
                # binarize mask
                gt_mask[gt_mask <= 10] = 0      # Black
                gt_mask[gt_mask > 10] = 1     # White
                #print('gt_mask',gt_mask.shape)
                #print('gt nonzero',np.count_nonzero(gt_mask))
                high, wid, channel = clip_list_temp[k][0].shape
                #print('shape',high, wid)
                gt_mask = resize(gt_mask, high, wid)
                if int(self.args.no_thre)==1:
                    min_thre = 0
                else:
                    min_thre = float(self.args.min_thre)#0.2
                if k==0:
                    try:

                        query_gt_mask=gt_mask
                    except:
                        #print('crop error',class_name,vid,name,area_list[k])
                        clip_list = [self.spatial_transform(clip) for clip in clip_list_temp]
                        return clip_list, zero_mask,0,1,sample.class_index
                if k==1:
                    try:
                        
                        key_gt_mask = gt_mask
                    except:
                        #print('crop error',class_name,vid,name,area_list[k])
                        clip_list = [self.spatial_transform(clip) for clip in clip_list_temp]
                        return clip_list, zero_mask,0,1,sample.class_index

                region = clip_list_temp[k]
                crop_T_max = region.contiguous()
                clip_list.append(crop_T_max)
            
        if load_mask==1:
            query_mask_list = []
            query_frame_list = []
            frame_path_list = []
            if 1:
                if '17' in str(self.args.experiment_dir):
                    mask_frame = mask_list[1].split('/')[-2]
                else:
                    mask_frame = mask_list[1].split('/')[-1]
                mask_frame_n = int(mask_frame.split('_')[-1].split('.')[0])
                query_mask_path = mask_list[0]
                query_mask_array = np.zeros((frame_length,224,224),dtype=np.uint8)
                for i in range(frame_length):
                    cur_num = str(t_crop_q[0]-mid_n+i).zfill(5)
                    frame_name = cur_num
                    #print(frame_name)
                    #print('query_mask_path',query_mask_path)
                    #print('mask_frame',mask_frame)
                    # new_path = query_mask_path.replace(mask_frame,frame_name)
                    # max_id = new_path.split('jpg')[-1].split('mask')[0][1:-1]
                    # all_mask_name = new_path.replace(max_id,'*')
                    if '17' in str(self.args.experiment_dir):
                    
                        
                        frame_path = '/export/optical-flow/data/DAVIS2019/DAVIS/JPEGImages/'+class_name+\
                        '/'+vid+'/'+frame_name+'.jpg'
                    else:
                        
                        frame_path = '/export/home/motiongrouping/DAVIS/JPEGImages/'+class_name+\
                        '/'+vid+'/'+frame_name+'.jpg'
                    #query_frame = np.array(Image.open(frame_path))[data['top1']:data['bot1'],data['left1']:data['right1'],:]
                    #query_frame = resize(query_frame, 224, 224)
                    #query_frame = np.expand_dims(query_frame, axis=0)
                    query_frame_list.append(frame_path)
                    
                    #new_path = '/export/home/RSPNet/meta_data/vgg_test/'+class_name+'/'+vid+'/'+cur_num+'.png'
                    """
                    try:
                        1==2
                        #print('get mask')
                        max_mask = np.array(Image.open(new_path).convert('L'))
                        query_mask = max_mask
                        query_mask = query_mask[data['top1']:data['bot1'],data['left1']:data['right1']]
                        query_mask = resize(query_mask, 224, 224)
                        query_mask[query_mask <= 128] = 0      # Black
                        query_mask[query_mask > 128] = 1     # White
                        
                    except:
                    """
                    #print('use middle q')
                    #max_mask = np.array(Image.open(key_mask_path).convert('L'))
                    query_mask = query_gt_mask#gt_masks[0]
                    
                    #query_mask[query_mask <= 128] = 0      # Black
                    #query_mask[query_mask > 128] = 1     # White
                    try:
                        query_mask = resize(query_mask, 224, 224)
                    except:
                        print('t1, t2, ltrb1, ltrb2', mid_frame_q, mid_frame_k, first_crop_coords, second_crop_coords, gt_masks[0].shape, query_mask.shape)
                    query_mask_array[i,:,:] = query_mask
                    
            key_mask_list = []
            key_frame_list = []
            if 1:
            #try:
                if '17' in str(self.args.experiment_dir):
                    mask_frame = mask_list[1].split('/')[-2]
                else:
                    mask_frame = mask_list[1].split('/')[-1]
                mask_frame_n = int(mask_frame.split('_')[-1].split('.')[0])
                #print('key_mask_frame',mask_frame)
                key_mask_path = mask_list[1]
                key_mask_array = np.zeros((frame_length,224,224),dtype=np.uint8)
                for i in range(frame_length):
                    cur_num = str(mask_frame_n-mid_n+i).zfill(5)
                    frame_name = cur_num
                    if '17' in str(self.args.experiment_dir):
                    
                        
                        frame_path = '/export/optical-flow/data/DAVIS2019/DAVIS/JPEGImages/'+class_name+\
                        '/'+vid+'/'+frame_name+'.jpg'
                    else:
                        
                        frame_path = '/export/home/motiongrouping/DAVIS/JPEGImages/'+class_name+\
                        '/'+vid+'/'+frame_name+'.jpg'
                    key_frame_list.append(frame_path)
                    #print(frame_name)
                    #print('query_mask_path',query_mask_path)
                    #print('mask_frame',mask_frame)
                    # new_path = key_mask_path.replace(mask_frame,frame_name)
                    # max_id = new_path.split('jpg')[-1].split('mask')[0][1:-1]
                    # all_mask_name = new_path.replace(max_id,'*')

                    # new_path = '/export/home/RSPNet/meta_data/vgg_test/'+class_name+'/'+vid+'/'+cur_num+'.png'
                    """
                    try:
                        1==2
                        #print('get mask')
                        max_mask = np.array(Image.open(new_path).convert('L'))
                        #print(class_name,vid,np.count_nonzero(max_mask))
                
                        key_mask = max_mask
                        key_mask = key_mask[data['top2']:data['bot2'],data['left2']:data['right2']]
                        key_mask = resize(key_mask, 224, 224)
                        key_mask[key_mask <= 128] = 0      # Black
                        key_mask[key_mask > 128] = 1     # White
                    except:
                    """
                    #print('use middle k ')
                    key_mask = key_gt_mask#gt_masks[1]
                    
                    try:
                        key_mask = resize(key_mask, 224, 224)
                    except:
                        print('key t1, t2, ltrb1, ltrb2', mid_frame_q, mid_frame_k, first_crop_coords, second_crop_coords, gt_masks[1].shape, key_mask.shape)
                    #key_mask[key_mask <= 128] = 0      # Black
                    #key_mask[key_mask > 128] = 1     # White
                    
                    key_mask_array[i,:,:] = key_mask
                    
                
                query_mask_array2 = np.expand_dims(query_mask_array, axis=0)
                key_mask_array2 = np.expand_dims(key_mask_array, axis=0)
                #print('query non zero',class_name,vid,np.count_nonzero(query_mask_array2),np.count_nonzero(query_mask_array))
                #print('key non zero',class_name,vid,np.count_nonzero(key_mask_array2),np.count_nonzero(key_mask_array))
                mask_list = [query_mask_array2,key_mask_array2]
                #print('query_mask_array',query_mask_array.dtype)
                #print('mask_list',mask_list)
                #if index==21:
                #    print('index',index)
                #    print('clip_frame_indices_list',clip_frame_indices_list)
                #print(class_name,vid,np.count_nonzero(query_mask_array2),np.count_nonzero(key_mask_array2))
                #info = [class_name,vid,data['mask1'],frame_path_list,data]
                data={'mask1':query_mask_path,'mask2':key_mask_path}
                frame_path_list = [query_frame_list,key_frame_list]
                info = [class_name,vid,0,frame_path_list,data] #info is useless
                
                stop4 = timeit.default_timer()
                if ver==1:
                    #print('mask looading Time: ', stop4 - stop3) 
                    print(class_name,vid,t_crop_q,t_crop_k,mid_frame_q,mid_frame_k)
                
                return clip_list, mask_list, load_mask, info,sample.class_index #clip_list
                #return clip_list, mask_list, load_mask, info,sample.class_index #clip_list = two video segment
        else:
            info = [class_name,vid,0,0,0]
            #print('zero_mask',zero_mask[0].dtype)
            return clip_list, zero_mask, 0, info,sample.class_index

    def __len__(self):
        return len(self.samples)


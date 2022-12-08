query_mask_list = []
        query_frame_list = []
        frame_path_list = []
        if 1:
            mask_frame = data['mask1'].split('/')[-2]
            mask_frame_n = int(mask_frame.split('_')[-1])
            query_mask_path = data['mask1']
        
            for i in range(frame_length):
                cur_num = str(mask_frame_n-mid_n+i+1).zfill(5)
                frame_name = 'image_'+cur_num
                #print(frame_name)
                #print('query_mask_path',query_mask_path)
                #print('mask_frame',mask_frame)
                new_path = query_mask_path.replace(mask_frame,frame_name)
                max_id = new_path.split('jpg')[-1].split('mask')[0][1:-1]
                all_mask_name = new_path.replace(max_id,'*')

                frame_path = '/export/home/data/vgg_sound-256/frame/test_split/'+class_name+\
                '/'+data['vid']+'/'+frame_name+'.jpg'
                query_frame = np.array(Image.open(frame_path))[data['top1']:data['bot1'],data['left1']:data['right1'],:]
                query_frame = resize(query_frame, 224, 224)
                query_frame = np.expand_dims(query_frame, axis=0)
                query_frame_list.append(query_frame)
                frame_path_list.append(frame_path)
                #print('all_mask_name',all_mask_name)
                if i!=0:
                    max_IOU = 0
                    if len(glob(all_mask_name))==0:
                        print('query error', query_mask_path)
                        break
                    for all_mask in glob(all_mask_name):
                        #print('all_mask',all_mask)
                        gt_mask = np.array(Image.open(all_mask).convert('L'))
                        #print(gt_mask)
                        intersection = np.count_nonzero(gt_mask*prev_masl)
                        union = np.count_nonzero(gt_mask+prev_masl)
                        if union==0:
                            continue
                        iou_score = intersection / union
                        #print(all_mask,iou_score)
                        if iou_score>max_IOU:
                            max_mask=gt_mask
                            max_mask_name = all_mask
                            max_IOU=iou_score
                            #print('change max')
                            
                    #print('max_mask',max_mask_name)
                    #print('Max IoU is %s' % max_IOU)
                    #if max_IOU>
                    prev_masl=max_mask
                else:
                    #print('new_path',new_path)
                    try:
                        max_mask = np.array(Image.open(new_path).convert('L'))
                        prev_masl = max_mask
                    except:
                        #print('use middle')
                        max_mask = np.array(Image.open(query_mask_path).convert('L'))
                        prev_masl = max_mask
                #print('max_id',max_id)
                #print('new_path',new_path) 
                
                #cv2.imwrite(temp_folder+data['vid']+"_query_mask.png", max_mask)
                query_mask = max_mask
                """
                query_mask=np.expand_dims(query_mask, axis=2)
                
                query_mask = MF.get_reference_crop_covering_mask(
                    query_mask,
                    reference_coords=first_crop_coords,
                    other_coords=(
                         []
                    ),
                )
                print('query_mask',query_mask)
                """
                query_mask = query_mask[data['top1']:data['bot1'],data['left1']:data['right1']]
                query_mask = resize(query_mask, 224, 224)
                #
                #cv2.imwrite(temp_folder+data['vid']+"_query_mask_crop"+cur_num+".png", max_mask)
                query_mask[query_mask <= 128] = 0      # Black
                query_mask[query_mask > 128] = 1     # White
                
                #max_mask_neg = 1-max_mask
                #max_mask_neg = max_mask_neg*150
                #max_mask_neg = np.expand_dims(max_mask_neg, axis=2)
                #print('mask area',np.count_nonzero(max_mask))
                #print(frame_path)
                #print('query_mask',query_mask.shape)
                query_mask=np.expand_dims(query_mask, axis=0)
                #query_mask = np.transpose(query_mask, axes=[2, 0, 1])
                query_mask_list.append(query_mask)
                key_clip = clip_list[0]
                #print('key_clip',key_clip.shape) #[32, 221, 262, 3])
                #print('max_mask',max_mask.shape) # max_mask (221, 262, 1)
                #masked_key = key_clip*max_mask
                #masked_key = masked_key+max_mask_neg
                #masked_key[masked_key == 0] = 255
                #transform = alb.Compose([alb.ToGray(p=0.2)])
                #masked_key = transform(masked_key)
                #print('masked_key',masked_key.shape)
                #Image.fromarray(masked_key).save(temp_folder+data['vid']+"_masked_query_crop"+cur_num+".png")
            #print('query complete')
        query_mask_array = np.vstack(query_mask_list)
        query_frame_array = np.vstack(query_frame_list)
        #print('query_mask_array',query_mask_array.shape)
        #except:
        #    print('query seg error',query_mask_path)
        key_mask_list = []
        if 1:
        #try:
            mask_frame = data['mask2'].split('/')[-2]
            mask_frame_n = int(mask_frame.split('_')[-1])
            #print('key_mask_frame',mask_frame)
            key_mask_path = data['mask2']
            for i in range(frame_length):
                cur_num = str(mask_frame_n-mid_n+i+1).zfill(5)
                frame_name = 'image_'+cur_num
                #print(frame_name)
                #print('query_mask_path',query_mask_path)
                #print('mask_frame',mask_frame)
                new_path = key_mask_path.replace(mask_frame,frame_name)
                max_id = new_path.split('jpg')[-1].split('mask')[0][1:-1]
                all_mask_name = new_path.replace(max_id,'*')

                if i!=0:
                    max_IOU = 0
                    if len(glob(all_mask_name))==0:
                        print('key error',key_mask_path)
                        break
                    for all_mask in glob(all_mask_name):
                        
                        gt_mask = np.array(Image.open(all_mask).convert('L'))
                        #print(gt_mask)
                        intersection = np.count_nonzero(gt_mask*prev_masl)
                        union = np.count_nonzero(gt_mask+prev_masl)
                        if union==0:
                            continue
                        iou_score = intersection / union
                        #print(all_mask,iou_score)
                        if iou_score>max_IOU:
                            max_mask=gt_mask
                            max_mask_name = all_mask
                            max_IOU=iou_score
                            #print('change max')
                            
                    #print('max_mask_key',max_mask_name)
                    #print('Max IoU is %s' % max_IOU)
                    #if max_IOU>
                    prev_masl=max_mask
                else:
                    try:
                        max_mask = np.array(Image.open(new_path).convert('L'))
                        prev_masl = max_mask
                    except:
                        #print('use middle')
                        max_mask = np.array(Image.open(key_mask_path).convert('L'))
                        prev_masl = max_mask
                #print('max_id',max_id)
                #print('new_path',new_path) 
                
                #cv2.imwrite(temp_folder+data['vid']+"_key_mask.png", max_mask)
                key_mask = max_mask
                """
                key_mask=np.expand_dims(key_mask, axis=2)
                key_mask = MF.get_reference_crop_covering_mask(
                    key_mask,
                    reference_coords=second_crop_coords,
                    other_coords=(
                         []
                    ),
                )
                """
                key_mask = key_mask[data['top2']:data['bot2'],data['left2']:data['right2']]
                key_mask = resize(key_mask, 224, 224)
                #
                #cv2.imwrite(temp_folder+data['vid']+"_key_mask_crop"+cur_num+".png", max_mask)
                #print('max_mask b',max_mask)
                #print('mask area b',np.count_nonzero(max_mask))
                key_mask[key_mask <= 128] = 0      # Black
                key_mask[key_mask > 128] = 1     # White
                #print('max_mask',max_mask)
                #max_mask_neg = 1-query_mask
                #max_mask_neg = max_mask_neg*150
                #max_mask_neg = np.expand_dims(max_mask_neg, axis=2)
                #print('mask area',np.count_nonzero(max_mask))
                #frame_path = '/export/home/data/vgg_sound-256/frame/test_split/'+class_name+\
                #'/'+data['vid']+'/'+frame_name+'.jpg'
                #print(frame_path)
                #key_frame = np.array(Image.open(frame_path))[data['top2']:data['bot2'],data['left2']:data['right2'],:]
                #print('key_frame',key_frame.shape)
                #print('max_mask',max_mask.shape)
                #max_mask=np.expand_dims(max_mask, axis=2)
                
                #key_mask = np.transpose(key_mask, axes=[2, 0, 1])
                key_mask=np.expand_dims(key_mask, axis=0)
                key_mask_list.append(key_mask)
                key_clip = clip_list[1]
                #masked_key = key_clip*max_mask
                #masked_key = masked_key+max_mask_neg
                #masked_key[masked_key == 0] = 255
                #transform = alb.Compose([alb.ToGray(p=0.2)])
                #masked_key = transform(masked_key)
                #print('masked_key',masked_key.shape)
                #Image.fromarray(masked_key).save(temp_folder+data['vid']+"_masked_key_crop"+cur_num+".png")
            #print('key complete')
        #except:
        #    print('key seg error',key_mask_path)
        key_mask_array = np.vstack(key_mask_list)
        #print('key_mask_array',key_mask_array.shape)
        query_mask_array = np.expand_dims(query_mask_array, axis=0)
        key_mask_array = np.expand_dims(key_mask_array, axis=0)
        mask_list = [query_mask_array,key_mask_array]
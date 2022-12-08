"""
Add apex
"""

import logging
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pyhocon import ConfigTree
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from pathlib import Path
from PIL import Image

from arguments import Args
from datasets.classification import DataLoaderFactoryV3
from datasets.classification import DataLoaderFactoryV6
from framework import utils
from framework.config import get_config, save_config
from framework.logging import set_logging_basic_config
from framework.meters.average import AverageMeter
from framework.metrics.classification import accuracy, binary_accuracy
from framework.utils.checkpoint import CheckpointManager
from moco import ModelFactory
from moco.builder_diffspeed_diffloss import Loss
from utils.moco import replace_moco_k_in_config
from grad_cam_s3d import GradCAM
import cv2
import albumentations as alb
import numpy as np
logger = logging.getLogger(__name__)
import optimizer

def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img = cv2.resize(img, (width, height), interpolation=interpolation)
    return img

class Engine:

    def __init__(self, args: Args, cfg: ConfigTree, local_rank: int):
        self.args = args
        self.cfg = cfg
        self.local_rank = local_rank

        self.opt_level = self.cfg.get_string('opt_level')
        #print('cfg',cfg)
        self.model_factory = ModelFactory(cfg)
        if int(self.args.temp)==1:
            print('Use temp')
            self.data_loader_factory = DataLoaderFactoryV6_temp(cfg,args)
        else:
            self.data_loader_factory = DataLoaderFactoryV6(cfg,args)

        self.device = torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        self.loss_lambda = self.cfg.get_config('loss_lambda')

        self.model = self.model_factory.build_moco_diffloss()
        self.criterion = Loss(
            margin=2.0,
            A=self.loss_lambda.get_float('A'),
            M=self.loss_lambda.get_float('M')
        )

        self.train_loader = self.data_loader_factory.build(vid=True, device=self.device)

        self.learning_rate = self.cfg.get_float('optimizer.lr')
        self.batch_size = self.cfg.get_int('batch_size')
        if not self.args.no_scale_lr:
            self.learning_rate = utils.environment.scale_learning_rate(
                self.learning_rate,
                self.args.world_size,
                self.batch_size,
            )
        if int(self.args.adamw)==1:
            #print('adamw')
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=.1,
            )
        elif int(self.args.lars)==1:
            #if args.optimizer == 'lars':
            self.optimizer = optimizer.LARS(self.model.parameters(), self.learning_rate,
                                        weight_decay=1e-6,
                                        momentum=0.9)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.cfg.get_float('optimizer.momentum'),
                dampening=self.cfg.get_float('optimizer.dampening'),
                weight_decay=self.cfg.get_float('optimizer.weight_decay'),
                nesterov=self.cfg.get_bool('optimizer.nesterov'),
            )

        self.num_epochs = cfg.get_int('num_epochs')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate / 1000
        )

        self.arch = cfg.get_string('arch')

        if local_rank == 0:
            self.summary_writer = SummaryWriter(
                log_dir=str(args.experiment_dir)
            )
            self.checkpoint = CheckpointManager(
                args.experiment_dir,
                keep_interval=cfg.get_int('checkpoint_interval')
            )
        else:
            self.summary_writer = None
            self.checkpoint = None

        self.log_interval = cfg.get_int('log_interval')

        self.loss_meter = AverageMeter('Loss', device=self.device)
        self.loss_meter_A = AverageMeter('Loss_A', device=self.device)  # Scale of this is large
        self.top1_meter_A = AverageMeter('Acc@1_A', fmt=':6.2f', device=self.device)
        self.top5_meter_A = AverageMeter('Acc@5_A', fmt=':6.2f', device=self.device)

        self.top1_meter_A_n = AverageMeter('Acc@1_A_n', fmt=':6.2f', device=self.device)
        self.top5_meter_A_n = AverageMeter('Acc@5_A_n', fmt=':6.2f', device=self.device)

        self.loss_meter_M = AverageMeter('Loss_M', device=self.device)  # Scale of this is large
        self.top1_meter_M = AverageMeter('Acc@1_M', fmt=':6.2f', device=self.device)

        self.current_epoch = 0

        self.overall_progress = None  # Place Holder

    def _load_ckpt_file(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=self.device)
        if states['arch'] != self.arch:
            raise ValueError(f'Loading checkpoint arch {states["arch"]} does not match current arch {self.arch}')
        return states

    def load_checkpoint(self, checkpoint_path):
        states = self._load_ckpt_file(checkpoint_path)
        logger.info('Loading checkpoint from %s', checkpoint_path)
        self.model.module.load_state_dict(states['model'])

        self.optimizer.load_state_dict(states['optimizer'])
        self.scheduler.load_state_dict(states['scheduler'])

        self.current_epoch = states['epoch']
        self.best_loss = states['best_loss']

    def load_model(self, checkpoint_path):
        states = self._load_ckpt_file(checkpoint_path)
        logger.info('Loading model from %s', checkpoint_path)
        self.model.module.load_state_dict(states['model'])

    def reset_meters(self):
        self.loss_meter.reset()

        self.loss_meter_A.reset()
        self.top1_meter_A.reset()
        self.top5_meter_A.reset()

        self.top1_meter_A_n.reset()
        self.top5_meter_A_n.reset()

        self.loss_meter_M.reset()
        self.top1_meter_M.reset()

    def train_epoch(self,gcam_q):
        import gc ; gc.collect()
        epoch = self.current_epoch
        self.train_loader.set_epoch(epoch)
        num_iters = len(self.train_loader)
        self.reset_meters()

        
        target_layer_q = "encoder_q.layer4" #+ args.layer_name
        


        iter_data = tqdm(self.train_loader, desc='Current Epoch', disable=self.local_rank != 0, dynamic_ncols=True)
        totoal_IOU = 0
        IOU_count=0
        crop_error = 0
        for i, ((clip_q, clip_k), (mask_q, mask_k), load_mask,info, *_) in enumerate(iter_data):
            
            #for i, ((clip_q, clip_k),  *_) in enumerate(iter_data):
            #print('clip_q',clip_q.shape)
            #print('clip_k',clip_k.shape)
            if self.args.onlycc:
                grad_loss = 0
                #print('Only CC')
            else:
                grad_loss = 1


            for j in range(len(info)):
                if info[j]==1:
                    print('crop error',crop_error)
                    crop_error+=1
            #"""
            #print('mask_q',mask_q.shape)
            #print('mask_k',mask_k.shape)
            output, target, ranking_logits, ranking_target = gcam_q.forward(clip_q, clip_k, add_to_queue=True)
            # start from RSP on VGG sound training
            _cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
            _margin_ranking_loss = nn.MarginRankingLoss(margin=2).to(self.device)

            ce1 = _cross_entropy_loss(output[0], target)
            ce2 = _cross_entropy_loss(output[1], target)
            ranking = _margin_ranking_loss(ranking_logits[0], ranking_logits[1], ranking_target)
            loss_s =  (ce1 + ce2) +  ranking
            loss_A = ce1 + ce2
            loss_M = ranking
            
            #loss_s, loss_A, loss_M = self.criterion(output, target, ranking_logits, ranking_target)
            #clip_q (batch x Time x channel x H x W)
            #mask_q (batch x Time x channel x H x W) 1/0
            #print('clip_k',clip_k.shape)
            #print('mask_k',mask_k.shape)
            #""" compute gradcam
            if grad_loss==1:# and load_mask==1:
                #print('clip_k',clip_k.dtype)
                #print('load_mask',load_mask)
                #print('mask_k',mask_k)
                #print('info',info)
                mask_k = mask_k.repeat(1,1,32,1,1)
                masked_key = clip_k*mask_k
                #print('masked_key',masked_key.dtype)
                #clip_q_c =  clip_q.clone()
                #output, target, ranking_logits, ranking_target = gcam_q.forward( clip_q, clip_k, add_to_queue=True)
                output_masked, target_masked,_,_2 = gcam_q.forward(clip_q, masked_key, add_to_queue=False)
                # compute gradients of the dot-product wrt query convolutional activations
                #"""
                grad_wrt_act_masked = gcam_q.backward(
                    torch.zeros(0, dtype=clip_q.dtype, device=self.device),
                    target_layer=target_layer_q
                )
                #print('grad_wrt_act_masked',grad_wrt_act_masked.shape) #[8, 1024, 2, 7, 7]
                # combine gradients with forward query activation maps to get gradcam
                gcam_masked = gcam_q.generate(
                    target_layer=target_layer_q, grads=grad_wrt_act_masked
                )
                #print('gcam_masked',gcam_masked.shape) #[8, 1, 2, 7, 7]

                output_mask_size = 7
                #print('mask_query',mask_q)
                #print('non zero',torch.nonzero(mask_q))
                # batch * 1 * Time * 224 * 224
                #Use middle frame
                mask_q_mid = mask_q[:,:,0,:,:] #using middle frame, may change to averaged frame
                #Use average feature
                #mask_q_mid = torch.mean(mask_q.type(torch.FloatTensor),2)
                
                mask_query_interp = torch.nn.functional.interpolate(mask_q_mid.type(torch.FloatTensor), \
                output_mask_size, mode='bilinear', align_corners=True).squeeze(0).to(self.device)
                criterion_gcam = nn.CosineSimilarity(dim=1).to(self.device)
                mask_query_interp = mask_query_interp.repeat(1,2,1,1)
                #print('mask_query_interp',mask_query_interp.shape) #16, 1, 7, 7
                #gcam_masked = torch.mean(gcam_masked,2)
                gcam_masked = gcam_masked.squeeze(1).squeeze(1)#.clone()
                mask_query_interp = mask_query_interp.squeeze(1)
                #print('gcam_masked',gcam_masked.shape) #[2, 1, 1, 7, 7]
                #print('mask_query_interp',mask_query_interp.shape) #([2, 1, 7, 7])
                #a = [0, 1, 0, 1, 0, 0, 0, 0]
                loss_gcam_masked_total =0
                #loss_gcam_len=0
                bs =len(load_mask)
                #load_mask = load_mask.to(self.device)
                load_mask = torch.as_tensor(load_mask)
                load_mask = load_mask.cuda(device=self.device, non_blocking=True)
                loss_gcam_masked = 1-criterion_gcam(gcam_masked.view(bs,-1), mask_query_interp.view(bs,-1))
                #print('shape',gcam_masked.shape,load_mask.shape)
                grad_sum = torch.dot(loss_gcam_masked,load_mask.float())
                
                loss_gcam_masked_total = grad_sum/torch.sum(load_mask)
                gw = float(self.args.grad_weight)
                
                loss = loss_s +  gw*loss_gcam_masked_total
                loss_M = gw*loss_gcam_masked_total
            else:
                loss = loss_s
            if not self.args.validate:
                gcam_q.model.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            #continue
            
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1_A, acc5_A = accuracy(output[0], target, topk=(1, 5))
            acc1_M, = accuracy(torch.cat(ranking_logits, dim=1), target, topk=(1,))

            acc1_A_n, acc5_A_n = accuracy(output[1], target, topk=(1, 5))
            batch_size = len(clip_q)
            self.top1_meter_A_n.update(acc1_A_n, batch_size)
            self.top5_meter_A_n.update(acc5_A_n, batch_size)

            if i > 0 and i % self.log_interval == 0:
                # Do logging as late as possible. this will force CUDA sync.
                # Log numbers from last iteration, just before update
                logger.info(
                    f'Train [{epoch}/{self.num_epochs}][{i - 1}/{num_iters}]'
                    f'\t{self.loss_meter_A}\t{self.top1_meter_A}\t{self.top5_meter_A}\n'
                    f'{self.loss_meter_M}\t{self.top1_meter_M}\n'
                    f'{self.top1_meter_A_n}\t{self.top5_meter_A_n}'
                )

            batch_size = len(clip_q)
            self.loss_meter.update(loss.detach(), batch_size)

            self.loss_meter_A.update(loss_A.detach(), batch_size)
            self.top1_meter_A.update(acc1_A, batch_size)
            self.top5_meter_A.update(acc5_A, batch_size)

            self.loss_meter_M.update(loss_M.detach(), batch_size)
            self.top1_meter_M.update(acc1_M, batch_size)

            self.overall_progress.update()

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                'train/loss', self.loss_meter.avg, epoch
            )

            self.summary_writer.add_scalar(
                'train/loss_A', self.loss_meter_A.avg, epoch
            )
            self.summary_writer.add_scalar(
                'train/acc1_A', self.top1_meter_A.avg, epoch
            )
            self.summary_writer.add_scalar(
                'train/acc5_A', self.top5_meter_A.avg, epoch
            )
            self.summary_writer.add_scalar(
                'train/loss_M', self.loss_meter_M.avg, epoch
            )
            self.summary_writer.add_scalar(
                'train/acc1_M', self.top1_meter_M.avg, epoch
            )
        #print('total IOU',totoal_IOU)
        torch.cuda.empty_cache()

    def run(self):

        num_epochs = 1 if self.args.debug else self.num_epochs
        best_loss = float('inf')
        with torch.autograd.set_detect_anomaly(True):
        #if 1:
            self.model.train()
            candidate_layers_q = ["encoder_q.layer4" ]
            gcam_q = GradCAM(model=self.model, candidate_layers=candidate_layers_q)
            num_iters = len(self.train_loader)

            with tqdm(total=num_epochs * num_iters,
                    disable=self.local_rank != 0,
                    smoothing=0.1,
                    desc='Overall',
                    dynamic_ncols=True,
                    initial=self.current_epoch * num_iters) as self.overall_progress:
                while self.current_epoch < num_epochs:
                    self.train_epoch(gcam_q)

                    self.scheduler.step()
                    if self.summary_writer is not None:
                        self.summary_writer.add_scalar('train/lr', utils.get_lr(self.optimizer), self.current_epoch)

                    self.current_epoch += 1

                    if self.local_rank == 0:
                        loss = self.loss_meter.avg.item()
                        is_best = loss < best_loss
                        best_loss = min(loss, best_loss)

                        self.checkpoint.save(
                            {
                                'epoch': self.current_epoch,
                                'arch': self.arch,
                                'model': self.model.module.state_dict(),
                                'best_loss': best_loss,
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                            },
                            is_best,
                            self.current_epoch,
                        )


def main_worker(local_rank: int, args: Args, dist_url: str):
    print('Local Rank:', local_rank)

    if args.seed is not None:
        utils.reproduction.initialize_seed(args.seed + local_rank)

    if local_rank == 0:
        set_logging_basic_config(args, tqdm=True)

    logger.info(f'Args = \n{args}')

    if args.config is None or args.experiment_dir is None:
        logger.error('No config or experiment_dir')

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=args.world_size,
        rank=local_rank,
    )

    utils.reproduction.cudnn_benchmark()

    cfg = get_config(args)

    replace_moco_k_in_config(cfg)

    if local_rank == 0:
        save_config(args, cfg)

    engine = Engine(args, cfg, local_rank=local_rank)
    if args.load_model is not None:
        engine.load_model(args.load_model)
    if args.load_checkpoint is not None:
        engine.load_checkpoint(args.load_checkpoint)

    if args.validate:
        # Only used to retrieve statistical results
        with torch.no_grad():
            with trange(1) as engine.overall_progress:
                engine.train_epoch()
    else:
        engine.run()


def main():
    args = Args.from_args()

    if args.debug:
        pass
    elif args.world_size < 2:
        warnings.warn('World size must be larger than 1')
        exit()

    if args.seed is not None:
        utils.reproduction.initialize_seed(args.seed)

    utils.environment.ulimit_n_max()

    # Run on main process to avoid conflict
    args.resolve_continue()
    args.make_run_dir()
    args.save()
    #utils.pack_code(args.run_dir)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'

    print(f'world_size={args.world_size} Using dist_url={dist_url}')

    args.parser = None
    # Only single node distributed training is supported
    mp.spawn(main_worker, args=(args, dist_url,), nprocs=args.world_size)


if __name__ == '__main__':
    main()

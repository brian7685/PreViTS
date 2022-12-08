import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
from typed_args import add_argument

from framework.arguments import Args as BaseArgs

logger = logging.getLogger(__name__)


def get_world_size() -> int:
    """
    It has to be larger than 2. Otherwise, the shuffle bn cannot work.
    :return:
    """
    num_gpus = torch.cuda.device_count()
    return max(2, num_gpus)
    # return num_gpus


@dataclass
class Args(BaseArgs):
    load_checkpoint: Optional[Path] = add_argument(
        '--load-checkpoint', required=False, type=Path,
        help='path to the checkpoint file to be loaded'
    )
    load_model: Optional[Path] = add_argument(
        '--load-model', required=False, type=Path, 
        help='path to the checkpoint file to be loaded, but only load model.'
    )
    validate: bool = add_argument(
        '--validate', action='store_true',
        help='Only run final validate then exit'
    )
    moco_checkpoint: Optional[str] = add_argument(
        '--mc', '--moco-checkpoint',
        help='load moco checkpoint'
    )
    onlycc: bool = add_argument(
        '--onlycc', action='store_true',
        help='Only CC'
    )
    lars: bool = add_argument(
        '--lars', action='store_true',
        help='Only CC'
    )
    interpo: bool = add_argument(
        '--interpo', action='store_true',
        help='Only CC'
    )
    adamw: bool = add_argument(
        '--adamw', action='store_true',
        help='Only CC'
    )
    
    random_c: bool = add_argument(
        '--random_c', action='store_true',
        help='Only CC'
    )
    with_flip: bool = add_argument(
        '--with_flip', action='store_true',
        help='Only CC'
    )
    random_con: bool = add_argument(
        '--random_con', action='store_true',
        help='Only CC'
    )
    no_thre: bool = add_argument(
        '--no_thre', action='store_true',
        help='Only CC'
    )
    center: bool = add_argument(
        '--center', action='store_true',
        help='Only CC'
    )
    ver: bool = add_argument(
        '--ver', action='store_true',
        help='Only CC'
    )
    salOnly: bool = add_argument(
        '--salOnly', action='store_true',
        help='Only CC'
    )
    nocc: bool = add_argument(
        '--nocc', action='store_true',
        help='Only CC'
    )
    noMaskMid: bool = add_argument(
        '--noMaskMid', action='store_true',
        help='Only CC'
    )
    noMask: bool = add_argument(
        '--noMask', action='store_true',
        help='Only CC'
    )
    max: bool = add_argument(
        '--max', action='store_true',
        help='Only CC'
    )
    bi: bool = add_argument(
        '--bi', action='store_true',
        help='Only CC'
    )
    seed: Optional[int] = add_argument(
        '--seed', help='random seed'
    )
    world_size: int = add_argument(
        '--ws', '--world-size', default=torch.cuda.device_count(),
        help='total processes'
    )
    mocoW: int = add_argument(
         '--mocoW', default=1,
        help='total processes'
    )
    vOut: str = add_argument(
        '--vOut', default='grad_track_train_result_multi_0',
        help='total processes'
    )
    split_type: str = add_argument(
        '--split_type', default='grad_track_train_result_multi_0',
        help='total processes'
    )
    method_type: str = add_argument(
        '--method_type', default='grad_track_train_result_multi_0',
        help='total processes'
    )
    multi: int = add_argument(
        '--multi',  default=0,
        help='total processes'
    )
    multi_v: bool = add_argument(
        '--multi_v', action='store_true',
        help='total processes'
    )
    light: bool = add_argument(
        '--li', action='store_true',
        help='total processes'
    )
    light_op: int = add_argument(
        '--liop', action='store_true',
        help='total processes'
    )
    time: int = add_argument(
        '--time', action='store_true',
        help='total processes'
    )
    mid_n: int = add_argument(
        '--mid_n',  default=16,
        help='total processes'
    )
    mid: int = add_argument(
        '--mid', action='store_true',
        help='total processes'
    )
    lamb: int = add_argument(
        '--lamb', action='store_true',
        help='total processes'
    )
    cast: int = add_argument(
        '--cast', action='store_true',
        help='total processes'
    )
    interval: int = add_argument(
         '--interval', default=90,
        help='total processes'
    )
    frame_no: int = add_argument(
         '--frame_no', default=-1,
        help='total processes'
    )
    split: int = add_argument(
         '--split', default=0,
        help='total processes'
    )
    withSeg_thre: int = add_argument(
         '--withSeg_thre', default=60,
        help='total processes'
    )
    temp: int = add_argument(
        '--tmp', '--temporal', default=0,
        help='total processes'
    )
    mid_w: float = add_argument(
        '--mid_w',  default=1.0,
        help='total processes'
    )
    
    seg_per: bool = add_argument(
        '--seg_per', action='store_true',
        help='total processes'
    )
    davis: bool = add_argument(
        '--davis', action='store_true',
        help='total processes'
    )
    constant: bool = add_argument(
        '--constant', action='store_true',
        help='total processes'
    )
    grad_weight: float = add_argument(
        '--gw', '--grad_weight', default=1.0,
        help='total processes'
    )
    
    min_thre: float = add_argument(
        '--min_thre', default=0.2,
        help='total processes'
    )
    lr: float = add_argument(
        '--lr', default=0.3,
        help='total processes'
    )
    trans_weight: float = add_argument(
        '--tw',  default=1.0,
        help='total processes'
    )
    
    single_layer: bool = add_argument(
        '--single_layer', action='store_true',
        help='total processes'
    )
    fixed_temp: bool = add_argument(
        '--fixT', action='store_true',
        help='total processes'
    )
    close: bool = add_argument(
        '--close', action='store_true',
        help='total processes'
    )
    rand_f: bool = add_argument(
        '--rand_f', action='store_true',
        help='total processes'
    )
    sal: bool = add_argument(
        '--sal', action='store_true',
        help='total processes'
    )
    _continue: bool = add_argument(
        '--continue', action='store_true',
        help='Use previous config and checkpoint',
    )
    no_scale_lr: bool = add_argument(
        '--no-scale-lr', action='store_true',
        help='Do not change lr according to batch size'
    )

    def resolve_continue(self):
        if not self._continue:
            return
        if not self.experiment_dir.exists():
            raise EnvironmentError(f'Experiment directory "{self.experiment_dir}" does not exists.')

        if self.config is None:
            run_id = -1
            for run in self.experiment_dir.iterdir():
                match = self.RUN_DIR_NAME_REGEX.match(run.name)
                if match is not None:
                    this_run_id = int(match.group(1))
                    if this_run_id > run_id and run.is_dir():
                        this_config_path = run / 'config.json'
                        if this_config_path.exists():
                            run_id = this_run_id
                            self.config = this_config_path
            if self.config is None:
                raise EnvironmentError(f'No previous run config found')
            logger.info('Continue using previous config: "%s"', self.config)
        if self.load_checkpoint is None:
            checkpoint_path = self.experiment_dir / 'checkpoint.pth.tar'
            if checkpoint_path.exists():
                self.load_checkpoint = checkpoint_path
                logger.info('Continue using previous checkpoint: "%s"', self.load_checkpoint)
            else:
                logger.warning('No previous checkpoint found')

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext

from datapipe.datasets import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.degradations import apply_mri_degradation

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import time
import wandb

def compute_nmse(pred, target):
    return F.mse_loss(pred, target) / torch.mean(target ** 2)

def compute_ncc(pred, target):
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    return torch.mean((pred - pred_mean) * (target - target_mean)) / (pred_std * target_std + 1e-8)

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        self.autoencoder = None

        # setup seed
        self.setup_seed()

        self.loss_mean = {}
        self.loss_count = None

        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

        # === W&B Setup for GPU Monitoring Only ===
        self.use_wandb = self.configs.train.get("use_wandb", False)
        if self.rank == 0 and self.use_wandb:
            wandb.init(
                project=self.configs.train.get("wandb_project", "gpu-monitoring"),
                name=self.configs.train.get("wandb_run_name", f"gpu_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=OmegaConf.to_container(self.configs, resolve=True),
                mode=self.configs.train.get("wandb_mode", "offline")
            )


        self.tic = time.time()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK',0])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK',0]) if num_gpus > 1 else 0


    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # Set base log directory
        if self.rank == 0:
            self.logger_log_dir = save_dir

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        if self.rank == 0:
            mode = 'validation' if self.configs.get('validate_only', False) else 'training'
            log_path = save_dir / f"{mode}.log"

            # clear any existing handlers
            self.logger = logger
            self.logger.remove()

            # write to file (overwrite for validation runs, append for training)
            file_mode = 'w' if mode == 'validation' else 'a'
            self.logger.add(
                log_path,
                format="{message}",
                mode=file_mode,
                level='INFO'
            )

            # always also log to stdout
            self.logger.add(sys.stdout, format="{message}")

            # record base dir and banner
            self.logger.log_dir = str(save_dir)
            print(f"[INFO] Logging to {log_path.name} ({mode} mode)")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            # ensure the logging directory and any missing parents exist
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))


        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float)
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            ema_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # image saving dirs
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            (image_dir / 'train').mkdir(parents=True, exist_ok=True)
            (image_dir / 'val').mkdir(parents=True, exist_ok=True)
            self.image_dir = image_dir

        # config dump
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))



    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

        if self.rank == 0 and self.use_wandb:
            wandb.finish()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt_state):
            """
            Merge ckpt_state into ema_state (both are state-dicts).
            For each key in ema_state, try 'module.<key>' then '<key>' in ckpt_state.
            """
            for key in list(ema_state.keys()):
                module_key = f"module.{key}"
                if module_key in ckpt_state:
                    ema_state[key] = ckpt_state[module_key].detach().clone()
                elif key in ckpt_state:
                    ema_state[key] = ckpt_state[key].detach().clone()
                else:
                    self.logger.warning(
                        f"EMA checkpoint missing '{key}' and '{module_key}', skipping."
                    )

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")

            # reload main model
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()

            if not self.configs.get('validate_only', False):
                # optimizer
                if hasattr(self, 'optimizer') and 'optimizer' in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                    if self.rank == 0:
                        self.logger.info("Optimizer state restored")

                # lr scheduler
                if hasattr(self, 'lr_scheduler') and 'lr_scheduler' in ckpt and ckpt['lr_scheduler'] is not None:
                    self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                    if self.rank == 0:
                        self.logger.info("LR scheduler state restored")

                # iteration counter
                self.iters_start = ckpt.get('iters_start', 0)

                # adjust LR up to current iters
                if hasattr(self, 'lr_scheduler'):
                    for ii in range(1, self.iters_start + 1):
                        self.adjust_lr(ii)

                # logging steps
                if self.rank == 0:
                    self.log_step     = ckpt.get('log_step',     self.log_step)
                    self.log_step_img = ckpt.get('log_step_img', self.log_step_img)

                # report resume info
                if self.rank == 0 and hasattr(self, 'optimizer'):
                    lr_now = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"Resuming from iteration {self.iters_start} with LR {lr_now:.2e}")
            else:
                # validation only mode
                self.iters_start = ckpt.get('iters_start', 0)

            # EMA state (if used)
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_" + Path(self.configs.resume).name)
                if ema_ckpt_path.exists():
                    self.logger.info(f"=> Loaded EMA checkpoint from {ema_ckpt_path}")
                    ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                    _load_ema_state(self.ema_state, ema_ckpt)
                    self.logger.info("EMA state restored")
                else:
                    self.logger.warning(f"EMA checkpoint not found at {ema_ckpt_path}")

            torch.cuda.empty_cache()

            # AMP scaler
            if not self.configs.get('validate_only', False):
                if hasattr(self, 'amp_scaler') and self.amp_scaler is not None and 'amp_scaler' in ckpt:
                    self.amp_scaler.load_state_dict(ckpt['amp_scaler'])
                    if self.rank == 0:
                        self.logger.info("AMP scaler restored from checkpoint")

            # reset seed to match resume point
            self.setup_seed(seed=self.iters_start)

        else:
            self.iters_start = 0




    def setup_optimizaton(self):
        model_params = list(self.model.parameters())

        if hasattr(self, 'autoencoder') and self.autoencoder is not None:
            ae_params = [p for p in self.autoencoder.parameters() if p.requires_grad]
            model_params += ae_params

        self.optimizer = torch.optim.AdamW(
            model_params,
            lr=self.configs.train.lr,
            weight_decay=self.configs.train.weight_decay
        )

        # AMP settings
        self.use_amp = self.configs.train.use_amp
        self.amp_scaler = amp.GradScaler(enabled=self.use_amp)

        # Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.configs.train.iterations,
            eta_min=self.configs.train.lr_min
        )

    def build_model(self):
        params = self.configs.model.get('params', dict)
        if self.configs.data['train']['type'] == 'mri':
            params['in_channels'] = 1  # Set input channels to 1 for grayscale MRI
            params['out_channels'] = 1  # Output should also be grayscale
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)  # wrap the network
        else:
            self.model = model

        # EMA
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        # Ensure MRI dataset is loaded correctly
        datasets = {}
        for phase in ['train', 'val']:
            if phase in self.configs.data:
                dataset_config = self.configs.data[phase]
                datasets[phase] = create_dataset(dataset_config)
                if self.rank == 0 and dataset_config['type'] == 'mri':
                    self.logger.info(f'Loaded {phase} MRI dataset with {len(datasets[phase])} images.')

        self.datasets = datasets

        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if 'val' in self.configs.data and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                    batch_size=self.configs.train.batch[1],
                                                    shuffle=False,
                                                    drop_last=False,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    )

            self.datasets = datasets
            self.dataloaders = dataloaders
            self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        """
        Prepares data for training/validation by applying necessary transformations.
        """
        for key, value in data.items():
            data[key] = value.cuda().to(dtype=dtype)

        if 'lq' in data and 'gt' in data:
            if self.configs.data[phase]['type'] == 'mri':  
                degradation_types = self.configs.data[phase].get('degradation_types', [])
                params = self.configs.data[phase].get('degradation_params', {})
                degraded = apply_mri_degradation(data['lq'], degradation_types=degradation_types, params=params)
                data['lq'] = degraded.clone().to(dtype=dtype)

        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()  # setup logger
        if self.rank == 0:
            self.logger.info("Training started")
        start_time = time.time()  # Start timing

        self.build_model()
        self.setup_optimizaton()
        self.resume_from_ckpt()
        self.build_dataloader()

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        total_time = 0.0
        log_interval = 10  # Log every 10 iterations

        for ii in range(self.iters_start, self.configs.train.iterations):
            start_iter_time = time.time()
            self.current_iters = ii + 1

            data = self.prepare_data(next(self.dataloaders['train']))
            self.training_step(data)

            if self.rank == 0:
                self.logger.info(f"[{ii + 1}/{self.configs.train.iterations}] Iteration complete")

            elapsed = time.time() - start_iter_time
            total_time += elapsed

            if (ii + 1) % log_interval == 0 and self.rank == 0:
                avg_time = total_time / log_interval
                remaining_iters = self.configs.train.iterations - (ii + 1)
                est_remaining = avg_time * remaining_iters
                mins, secs = divmod(est_remaining, 60)
                self.logger.info(f"Avg Time/Iter: {avg_time:.2f}s | Est. Time Left: {int(mins)}m {int(secs)}s")
                total_time = 0.0

            if 'val' in self.dataloaders and (ii + 1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            self.adjust_lr()

            if (ii + 1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if (ii + 1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii + 1)

        self.close_logger()

        if self.rank == 0:
            duration = time.time() - start_time
            mins, secs = divmod(duration, 60)
            self.logger.info(f"Training finished in {int(mins)}m {int(secs)}s")

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            #  Save main model checkpoint 
            ckpt_path = self.ckpt_dir / f'model_{self.current_iters:d}.pth'
            ckpt = {
                'iters_start': self.current_iters,
                'log_step': {phase: self.log_step[phase] for phase in ['train', 'val']},
                'log_step_img': {phase: self.log_step_img[phase] for phase in ['train', 'val']},
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)

            #  Save EMA model if used 
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / f'ema_model_{self.current_iters:d}.pth'
                torch.save(self.ema_state, ema_ckpt_path)

            #  Save Autoencoder weights (only if an autoencoder exists) 
            if getattr(self, 'autoencoder', None) is not None:
                ae_ckpt_path = self.ckpt_dir / f'autoencoder_{self.current_iters:d}.pth'
                torch.save(self.autoencoder.state_dict(), ae_ckpt_path)
                self.logger.info(f"Autoencoder saved to: {ae_ckpt_path}")
            else:
                self.logger.debug("Skipping autoencoder save (no autoencoder set).")


    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8, custom_suffix=""):
        """
        Save visual output images during training or validation.

        Args:
            im_tensor: Tensor of shape [B, C, H, W]
            tag: e.g., 'train_output', 'gt', 'lq'
            phase: 'train' or 'val'
            add_global_step: if True, increments step counter
            nrow: how many images to display per row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True)  # C x H x W

        if self.local_logging:
            save_dir = self.image_dir / phase / tag
            save_dir.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
            im_path = save_dir / f"{self.log_step_img[phase]}{custom_suffix}.png"
            im_np = im_tensor.cpu().permute(1, 2, 0).numpy()
            util_image.imwrite(im_np, str(im_path))

        if self.tf_logging:
            self.writer.add_image(
                f"{phase}/{tag}",
                im_tensor,
                self.log_step_img[phase] if add_global_step else 0,
            )

        if add_global_step:
            self.log_step_img[phase] += 1


    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                scalar_metrics = {
                    k: v.item() if isinstance(v, torch.Tensor) and v.ndim == 0
                    else float(v.mean()) if isinstance(v, torch.Tensor)
                    else float(v)
                    for k, v in metrics.items()
                }
                self.writer.add_scalars(tag, scalar_metrics, self.log_step[phase])
            else:
                if isinstance(metrics, torch.Tensor):
                    metrics = metrics.item() if metrics.ndim == 0 else metrics.mean().item()
                self.writer.add_scalar(tag, metrics, self.log_step[phase])

            if add_global_step:
                self.log_step[phase] += 1

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')

class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        # === Collect model parameters ===
        model_params = list(self.model.parameters())

        # === Collect trainable autoencoder parameters (if any) ===
        if self.autoencoder is not None:
            ae_trainable = [p for p in self.autoencoder.parameters() if p.requires_grad]
        else:
            ae_trainable = []

        total_params = model_params + ae_trainable

        # === Initialize optimizer ===
        self.optimizer = torch.optim.AdamW(
            total_params,
            lr=self.configs.train.lr,
            weight_decay=self.configs.train.weight_decay
        )

        # === Setup LR scheduler ===
        if self.configs.train.lr_schedule == 'constant':
            self.lr_scheduler = None
        elif self.configs.train.lr_schedule == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                eta_min=self.configs.train.lr_min,
            )
        elif self.configs.train.lr_schedule == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.configs.train.lr_min / self.configs.train.lr,
                total_iters=self.configs.train.iterations - self.configs.train.warmup_iterations,
            )
        elif self.configs.train.lr_schedule == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.configs.train.lr_step_size,
                gamma=self.configs.train.lr_gamma           
            )
        else:
            raise ValueError(f"Unknown LR scheduler: {self.configs.train.lr_schedule}")

        # === Logging param count ===
        if self.rank == 0:
            total_trainable = sum(p.numel() for p in total_params if p.requires_grad)
            self.logger.info(f"🔧 Optimizer initialized with {total_trainable / 1e6:.2f}M trainable parameters")

        # === AMP scaler ===
        self.use_amp = self.configs.train.use_amp
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def build_model(self):
        super().build_model()

        # === Autoencoder Setup ===
        if hasattr(self.configs, 'autoencoder') and self.configs.autoencoder is not None:
            ae_params = dict(self.configs.autoencoder.params)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**ae_params).cuda()

            # Load checkpoint
            if self.configs.autoencoder.ckpt_path:
                ae_ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
                if 'state_dict' in ae_ckpt:
                    ae_ckpt = ae_ckpt['state_dict']
                util_net.reload_model(autoencoder, ae_ckpt)

            if self.configs.autoencoder.tune_decoder:
                # === Finetune decoder only ===
                for k, v in autoencoder.named_parameters():
                    if 'decoder' in k or 'post_quant_conv' in k:
                        v.requires_grad = True
                    else:
                        v.requires_grad = False

                if self.rank == 0:
                    num_params = sum(
                        v.numel() for k, v in autoencoder.named_parameters()
                        if 'decoder' in k or 'post_quant_conv' in k
                    )
                    self.logger.info(f"Finetuning Decoder module: {num_params / 1e6:.2f}M parameters")

            else:
                # === Train full autoencoder ===
                for k, v in autoencoder.named_parameters():
                    v.requires_grad = True
                if self.rank == 0:
                    total_params = sum(p.numel() for p in autoencoder.parameters())
                    self.logger.info(f"Training full autoencoder: {total_params / 1e6:.2f}M parameters")

            autoencoder.train()

            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")

            self.autoencoder = autoencoder
            # === DEBUG: Check requires_grad for autoencoder ===
            if self.configs.get("debug", False):
                if self.autoencoder is not None:
                    for name, param in self.autoencoder.named_parameters():
                        print(f"[DEBUG] Autoencoder param: {name}, requires_grad={param.requires_grad}")
        else:
            self.autoencoder = None

        # === EMA Ignore Keys for Swin Models ===
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        # === LPIPS Metric ===
        lpips_net = getattr(self.configs.lpips, 'net', 'vgg') if hasattr(self.configs, 'lpips') else 'vgg'
        if self.rank == 0:
            self.logger.info(f"Loading LPIPS Metric: {lpips_net}...")

        lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}")
        for param in lpips_loss.parameters():
            param.requires_grad_(False)
        lpips_loss.eval()

        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")

        self.lpips_loss = lpips_loss

        # === Diffusion Setup ===
        diffusion_params = dict(self.configs.diffusion.params)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**diffusion_params)



    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.configs.degradation.sf, int):
                sf = self.configs.degradation.sf
            else:
                assert len(self.configs.degradation.sf) == 2
                sf = random.uniform(*self.configs.degradation.sf)

            if self.configs.degradation.use_sharp:
                im_gt = self.use_sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.configs.degradation['second_order_prob']:
                # blur
                if random.random() < self.configs.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.configs.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.configs.degradation['gray_noise_prob2']
                if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.configs.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.configs.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # resize back
            if self.configs.degradation.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
                temp_sf = self.configs.degradation['sf']
            else:
                temp_sf = self.configs.degradation['sf']

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.configs.degradation['gt_size']
            im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        elif phase == 'val':
            offset = self.configs.train.get('val_resolution', 256)
            for key, value in data.items():
                h, w = value.shape[2:]
                if h > offset and w > offset:
                    h_end = int((h // offset) * offset)
                    w_end = int((w // offset) * offset)
                    data[key] = value[:, :, :h_end, :w_end]
                else:
                    h_pad = math.ceil(h / offset) * offset - h
                    w_pad = math.ceil(w / offset) * offset - w
                    padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                    data[key] = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        else:
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        """
        Runs one backward pass, computing composite loss from MSE, SSIM, LPIPS, and TV,
        then stepping the optimizer (with AMP support).
        """
        context = torch.cuda.amp.autocast if self.use_amp else nullcontext
        with context():
            # 1. get raw losses and predictions from the diffusion wrapper
            losses, z_t, z0_pred = dif_loss_wrapper()

            # 2. upsample prediction if needed
            if self.configs.data['train']['type'] == 'mri':
                z0_pred_up = F.interpolate(
                    z0_pred,
                    size=micro_data['gt'].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                z0_pred_up = z0_pred

            # 3. remap to [0,1]
            pred_01 = (z0_pred_up + 1.0) * 0.5
            gt_01   = (micro_data['gt'] + 1.0) * 0.5

            # 4. compute true metrics on [0,1]
            ssim_val = self.ssim_loss(pred_01, gt_01).mean()
            mse_val  = F.mse_loss(pred_01, gt_01, reduction='none').mean()

            # 5. fetch optional LPIPS & TV (if dif_loss_wrapper provided them)
            lpips_val = losses.get("lpips", None)
            if lpips_val is not None:
                lpips_val = lpips_val.mean()
            tv_val = losses.get("tv", None)
            if tv_val is not None:
                tv_val = tv_val.mean()

            # 6. store individual terms for logging
            losses["ssim"] = ssim_val
            losses["mse"]  = mse_val
            if lpips_val is not None:
                losses["lpips"] = lpips_val
            if tv_val is not None:
                losses["tv"]    = tv_val

            # 7. build composite loss using config weights
            w = self.configs.train.loss_weights
            composite = w.mse   * mse_val \
                    + w.ssim  * (1.0 - ssim_val)
            if lpips_val is not None:
                composite += w.lpips * lpips_val
            if tv_val is not None:
                composite += w.tv    * tv_val

            losses["loss"] = composite

            # 8. final scalar loss (with gradient accumulation divisor)
            loss = composite.mean() / num_grad_accumulate

            # 9. rank-0 debug logging of latent ranges
            if self.rank == 0:
                self.logger.info(
                    f"[{self.current_iters}] z_start range: "
                    f"{z0_pred.min().item():.2f}→{z0_pred.max().item():.2f}"
                )
                self.logger.info(
                    f"[{self.current_iters}] pred_upsampled range: "
                    f"{z0_pred_up.min().item():.2f}→{z0_pred_up.max().item():.2f}"
                )

        # 10. zero grads and backprop
        self.optimizer.zero_grad()

        if self.use_amp:
            scaled_loss = self.amp_scaler.scale(loss)
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if any(p.grad is not None for p in self.model.parameters() if p.requires_grad):
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                print("Skipping optimizer.step(): no gradients found (AMP)")
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if any(p.grad is not None for p in self.model.parameters() if p.requires_grad):
                self.optimizer.step()
            else:
                print("Skipping optimizer.step(): no gradients found (non-AMP)")

        return losses, z0_pred_up, z_t


    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        # === Freeze autoencoder first 2000 iters ===
        #if self.current_iters < 2000:
            #for param in self.autoencoder.parameters():
                #param.requires_grad = False
            #if self.current_iters == 1999 and self.rank == 0:
                #self.logger.info("==> Unfreezing autoencoder at iter 2000.")
        #else:
            #for param in self.autoencoder.parameters():
                #param.requires_grad = True

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {
                key: value[jj:jj + micro_batchsize] for key, value in data.items()
            }
            last_batch = (jj + micro_batchsize >= current_batchsize)

            tt = torch.randint(
                0, self.base_diffusion.num_timesteps,
                size=(micro_data['gt'].shape[0],),
                device=f"cuda:{self.rank}",
            )

            latent_downsamping_sf = 2 ** (len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf

            if 'autoencoder' in self.configs:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]

            noise = torch.randn(
                size=(micro_data['gt'].shape[0], noise_chn, latent_resolution, latent_resolution),
                device=micro_data['gt'].device,
            )

            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq': micro_data['lq']}
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                model_kwargs = None

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )

            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            # Handle training metrics and visual logs only after final microbatch
            if last_batch:
                # Upsample z0_pred for consistent metrics and logging
                z0_pred_up = F.interpolate(z0_pred.detach(), size=micro_data['gt'].shape[-2:], mode='bilinear', align_corners=False)

                self.log_step_train(losses, tt, micro_data, z_t, z0_pred_up)

                pred_norm = (z0_pred_up + 1.0) * 0.5
                gt_norm   = (micro_data['gt'] + 1.0) * 0.5
                psnr_value = util_image.batch_PSNR(pred_norm, gt_norm)

                if self.rank == 0:
                    loss_total = losses["loss"].mean().item()
                    ssim_val = losses.get("ssim", torch.tensor(0.0)).mean().item()
                    tv_val = losses.get("tv_raw", torch.tensor(0.0)).mean().item()
                    lpips_val = losses.get("lpips", torch.tensor(0.0)).mean().item()
                    self.logger.info(
                        f'Train Step: Iter {self.current_iters} | '
                        f'Loss: {loss_total:.4f} | '
                        f'PSNR: {psnr_value:.2f} dB | '
                        f'LPIPS: {lpips_val:.4f} | '
                        f'SSIM: {ssim_val:.4f} | '
                        f'TV: {tv_val:.5f}'
                    )

                ssim_str = f"_SSIM{losses.get('ssim', torch.tensor(0.0)).mean().item():.4f}"
                lpips_str = f"_LPIPS{losses.get('lpips', torch.tensor(0.0)).mean().item():.4f}"
                mse_str   = f"_MSE{losses.get('mse', torch.tensor(0.0)).mean().item():.4f}"
                tv_str    = f"_TV{losses.get('tv', torch.tensor(0.0)).mean().item():.5f}" if 'tv' in losses else ""

                custom_suffix = f"_iter{self.current_iters}{ssim_str}{lpips_str}{mse_str}{tv_str}"

                if self.rank == 0 and self.configs.train.local_logging and \
                        self.current_iters % self.configs.train.log_freq[1] == 0:

                    self.logging_image(
                        z0_pred_up, tag='train_output', phase='train', add_global_step=True, custom_suffix=custom_suffix
                    )
                    self.logging_image(
                        micro_data['gt'], tag='gt', phase='train', add_global_step=False
                    )
                    self.logging_image(
                        micro_data['lq'], tag='lq', phase='train', add_global_step=False
                    )

            self.model.zero_grad()

            if hasattr(self.configs.train, 'ema_rate'):
                self.update_ema_model()

            # === GPU Usage Checker ===
            gpu_mem_allocated = torch.cuda.memory_allocated(device=f"cuda:{self.rank}") / (1024 ** 3)
            gpu_mem_reserved  = torch.cuda.memory_reserved(device=f"cuda:{self.rank}") / (1024 ** 3)

            if self.rank == 0 and self.current_iters % 100 == 0:
                self.logger.info(
                    f'[GPU Memory] Allocated: {gpu_mem_allocated:.2f} GB | Reserved: {gpu_mem_reserved:.2f} GB'
                )

                if self.use_wandb:
                    wandb.log({
                        'GPU/Memory Allocated (GB)': gpu_mem_allocated,
                        'GPU/Memory Reserved (GB)': gpu_mem_reserved,
                        'Iteration': self.current_iters
                    })

            
            # === Periodic optimizer state reset to avoid gradient accumulation issues ===
            if self.current_iters > 0 and self.current_iters % 2500 == 0:
                if self.rank == 0:
                    self.logger.info(
                        f"[Optimizer Reset] Iteration {self.current_iters}: Zeroing optimizer state (exp_avg and exp_avg_sq) to prevent gradient accumulation issues."
                    )
                num_reset = 0
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        state = self.optimizer.state[p]
                        if 'exp_avg' in state:
                            state['exp_avg'].zero_()
                            num_reset += 1
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'].zero_()
                            num_reset += 1
                if self.rank == 0:
                    self.logger.info(
                        f"[Optimizer Reset] Iteration {self.current_iters}: Zeroed optimizer state for {num_reset} parameter tensors."
                    )


    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        """
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        """
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]

            # Initialize running averages
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {
                    key: torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                    for key in loss.keys()
                }
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)

            # Accumulate losses at key timesteps
            for jj, step in enumerate(record_steps):
                index = step - 1
                mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                for key, value in loss.items():
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            # Log loss curves
            if self.current_iters % self.configs.train.log_freq[0] == 0:
                # avoid divide-by-zero
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4

                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count

                # Build and emit log string
                log_str = (
                    f"Train: {self.current_iters:06d}/{self.configs.train.iterations:06d}, "
                    f"Loss/MSE: "
                )
                for jj, step in enumerate(record_steps):
                    log_str += (
                        f"t({step}):{self.loss_mean['loss'][jj].item():.1e}/"
                        f"{self.loss_mean['mse'][jj].item():.1e}, "
                    )
                log_str += f"lr:{self.optimizer.param_groups[0]['lr']:.2e}"
                self.logger.info(log_str)

                # Send loss to TensorBoard / logger
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)

            # Log images and train‐time SSIM/PSNR
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                # Low‐quality and ground‐truth
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)

                # Diffused intermediate
                x_t = self.base_diffusion.decode_first_stage(
                    self.base_diffusion._scale_input(z_t, tt),
                    self.autoencoder,
                )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)

                # Model prediction
                x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                )
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

                # ——— Compute & log SSIM/PSNR on train batch ———
                # Remap from [-1,1] to [0,1]
                pred_01 = (x0_pred + 1.0) * 0.5
                gt_01   = (batch['gt'] + 1.0) * 0.5

                # Compute metrics
                ssim_val = self.ssim_loss(pred_01, gt_01).item()
                psnr_val = util_image.batch_PSNR(pred_01, gt_01)

                # Log to console
                self.logger.info(
                    f"Train @ iter {self.current_iters}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB"
                )
                # Log to TensorBoard / logger
                self.logging_metric(
                    {'ssim': [ssim_val], 'psnr': [psnr_val]},
                    tag='Metric',
                    phase=phase,
                    add_global_step=True
                )
                # ——————————————————————————————————————————————

            # Timing logs
            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elapsed = self.toc - self.tic
                self.logger.info(f"Elapsed time: {elapsed:.2f}s")
                self.logger.info("=" * 100)

    def validation(self, phase='val'):
        if self.rank == 0:
            import time
            start_time = time.time()  # ⏱ START TIMER

            print("\nStarting validation...")  

            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

            indices = np.linspace(
                0, self.base_diffusion.num_timesteps,
                self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                endpoint=False,
                dtype=np.int64,
            ).tolist()
            if not (self.base_diffusion.num_timesteps - 1) in indices:
                indices.append(self.base_diffusion.num_timesteps - 1)

            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)

            total_samples = 0
            ssim_scores, psnr_scores, nmse_scores, ncc_scores = [], [], [], []

            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')

                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']

                num_iters = 0
                if self.configs.model.params.cond_lq:
                    model_kwargs = {'lq': data['lq']}
                    if 'mask' in data:
                        model_kwargs['mask'] = data['mask']
                else:
                    model_kwargs = None

                tt = torch.tensor(
                    [self.base_diffusion.num_timesteps] * im_lq.shape[0],
                    dtype=torch.int64,
                ).cuda()

                for sample in self.base_diffusion.p_sample_loop_progressive(
                    y=im_lq,
                    model=self.ema_model if self.configs.train.use_ema_val else self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=True if self.autoencoder is None else False,
                    model_kwargs=model_kwargs,
                    device=f"cuda:{self.rank}",
                    progress=False,
                ):
                    if num_iters in indices:
                        sample_decode = {}
                        for key, value in sample.items():
                            if key in ['sample']:
                                decoded = self.base_diffusion.decode_first_stage(value, self.autoencoder).clamp(-1.0, 1.0)
                                sample_decode[key] = decoded
                                im_sr_progress = sample_decode['sample']
                                if num_iters + 1 == 1:
                                    im_sr_all = im_sr_progress
                                else:
                                    im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    num_iters += 1
                    tt -= 1

                if 'gt' in data:
                    sr_upsampled = F.interpolate(sample_decode['sample'], size=im_gt.shape[-2:], mode='bilinear', align_corners=False)
                    total_samples += im_gt.shape[0]

                    sr_norm = (sr_upsampled + 1.0) / 2.0
                    gt_norm = (im_gt + 1.0) / 2.0

                    ssim_val = ssim_metric(sr_norm, gt_norm).item()
                    psnr_val = psnr_metric(gt_norm.squeeze().cpu().numpy(), sr_norm.squeeze().cpu().numpy(), data_range=1.0)
                    nmse_val = compute_nmse(sr_norm, gt_norm).item()
                    ncc_val = compute_ncc(sr_norm, gt_norm).item()

                    ssim_scores.append(ssim_val)
                    psnr_scores.append(psnr_val)
                    nmse_scores.append(nmse_val)
                    ncc_scores.append(ncc_val)

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii + 1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(im_sr_all, tag='progress', phase=phase, add_global_step=False, nrow=len(indices))

                    # Log the final SR output as its own image
                    final_sr = sample_decode['sample']
                    final_sr = F.interpolate(final_sr, size=im_gt.shape[-2:], mode='bilinear', align_corners=False)
                    self.logging_image(final_sr, tag='sr_final', phase=phase, add_global_step=True)

                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                        self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

            if total_samples > 0:
                ssim_mean = np.mean(ssim_scores)
                psnr_mean = np.mean(psnr_scores)
                nmse_mean = np.mean(nmse_scores)
                ncc_mean = np.mean(ncc_scores)

                print(f"\nValidation Results over {total_samples} samples:")
                print(f"SSIM: {ssim_mean:.4f} | PSNR: {psnr_mean:.2f} dB | NMSE: {nmse_mean:.4f} | NCC: {ncc_mean:.4f}")
                self.logger.info(f"Validation Metrics - SSIM: {ssim_mean:.4f} | PSNR: {psnr_mean:.2f} | NMSE: {nmse_mean:.4f} | NCC: {ncc_mean:.4f}")

                self.logging_metric(ssim_mean, tag='SSIM', phase=phase, add_global_step=True)
                self.logging_metric(psnr_mean, tag='PSNR', phase=phase, add_global_step=True)
                self.logging_metric(nmse_mean, tag='NMSE', phase=phase, add_global_step=True)
                self.logging_metric(ncc_mean, tag='NCC', phase=phase, add_global_step=True)

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f" Validation took {mins}m {secs}s")
            self.logger.info(f"Validation time: {mins}m {secs}s")

            self.logger.info("=" * 100)
            print("Validation complete.\n")

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

class TrainerDifIRLPIPS(TrainerDifIR):

    def get_dynamic_loss_weights(self):
        """
        Interpolates loss weights between the 'initial' and 'final'
        values specified in configs.train.loss_weights over
        configs.train.loss_weights.ramp_iters iterations.
        Falls back to static defaults if the section is missing.
        """
        lw_cfg = getattr(self.configs.train, "loss_weights", None)
        if lw_cfg is None:
            return {"mse": 1.0, "lpips": 1.0, "ssim": 0.0, "tv": 0.0}

        init  = lw_cfg.initial
        final = lw_cfg.final
        ramp_iters = getattr(lw_cfg, "ramp_iters", 5000)
        t = min(1.0, self.current_iters / max(1, ramp_iters))

        return {
            "mse"  : init.mse  + (final.mse  - init.mse)  * t,
            "lpips": init.lpips+ (final.lpips- init.lpips)* t,
            "ssim" : init.ssim + (final.ssim - init.ssim) * t,
            "tv"   : init.tv   + (final.tv   - init.tv)   * t,
        }
    
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        weights = self.get_dynamic_loss_weights()
        w_mse = weights["mse"]
        w_lpips = weights["lpips"]
        w_ssim = weights["ssim"]
        w_tv = weights["tv"]

        if self.rank == 0 and self.current_iters % 100000 == 0:
            self.logger.info(
                f"Dynamic weights @ iter {self.current_iters}: "
                f"MSE={w_mse:.3f} | LPIPS={w_lpips:.3f} | SSIM={w_ssim:.3f} | TV={w_tv:.5f}"
            )

        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext

        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(z0_pred, self.autoencoder)
            self.current_x0_pred = x0_pred.detach()

            # Upsample latent prediction to match ground truth
            z0_pred_up = F.interpolate(
                z0_pred, size=micro_data['gt'].shape[-2:], mode='bilinear', align_corners=False
            )

            # Normalize for SSIM
            z0_pred_up_norm = (z0_pred_up + 1.0) / 2.0
            gt_norm = (micro_data['gt'] + 1.0) / 2.0

            # SSIM and MSE as true metrics
            ssim_val = self.ssim_loss(z0_pred_up_norm, gt_norm)
            mse_val = (z0_pred_up - micro_data['gt']).pow(2).mean()
            losses["ssim"] = ssim_val
            losses["mse"] = mse_val

            # Total Variation
            tv_val = total_variation_loss(z0_pred_up)
            losses["tv_raw"] = tv_val  # for logging only
            losses["tv"] = tv_val * w_tv  # weighted for training

            # LPIPS
            x0_clamped = x0_pred.clamp(-1.0, 1.0)
            gt_clamped = micro_data['gt'].clamp(-1.0, 1.0)
            lpips_val = self.lpips_loss(x0_clamped, gt_clamped).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(lpips_val))
            if flag_nan:
                lpips_val = torch.nan_to_num(lpips_val, nan=0.0)
            losses["lpips"] = lpips_val.mean() * w_lpips

            # Final composite loss
            losses["loss"] = (
                losses["mse"] * w_mse +
                losses["lpips"] +
                (1.0 - ssim_val) * w_ssim +
                losses["tv"]
            )

            loss = losses["loss"].mean() / num_grad_accumulate

        self.optimizer.zero_grad()

        if self.amp_scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if any(p.grad is not None for p in self.model.parameters() if p.requires_grad):
                self.optimizer.step()
        else:
            self.amp_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if any(p.grad is not None for p in self.model.parameters() if p.requires_grad):
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()

        return losses, z0_pred_up, z_t


    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]

            # Initialize rolling stats
            if not self.loss_mean or self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {
                    key: torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                    for key in loss.keys()
                }
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)

            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = (tt == index).float()

                    if value.dim() == 0:
                        current_loss = value.detach().item()
                    else:
                        while mask.dim() < value.dim():
                            mask = mask.unsqueeze(-1)
                        try:
                            mask = mask.expand_as(value)
                            current_loss = torch.sum(value.detach() * mask).item()
                            self.loss_count[jj] += mask.sum().item()
                        except RuntimeError:
                            current_loss = value.detach().sum().item()

                    self.loss_mean[key][jj] += current_loss

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count

                log_str = 'Train: {:06d}/{:06d}, MSE/LPIPS: '.format(
                    self.current_iters,
                    self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                        current_record,
                        self.loss_mean['mse'][jj].item() if 'mse' in self.loss_mean else 0,
                        self.loss_mean['lpips'][jj].item() if 'lpips' in self.loss_mean else 0,
                    )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)

                # Log all losses to TensorBoard
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)

                if 'lpips' in self.loss_mean:
                    self.logging_metric({'LPIPS': self.loss_mean['lpips']}, tag='Loss Components', phase=phase, add_global_step=True)
                if 'mse' in self.loss_mean:
                    self.logging_metric({'MSE': self.loss_mean['mse']}, tag='Loss Components', phase=phase, add_global_step=True)
                if 'ssim' in self.loss_mean:
                    self.logging_metric({'SSIM': self.loss_mean['ssim']}, tag='Loss Components', phase=phase, add_global_step=True)
                if 'tv' in self.loss_mean:
                    self.logging_metric({'TV': self.loss_mean['tv']}, tag='Loss Components', phase=phase, add_global_step=True)
                if 'loss' in self.loss_mean:
                    self.logging_metric({'Total Loss': self.loss_mean['loss']}, tag='Loss Components', phase=phase, add_global_step=True)

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logging_metric({'lr': current_lr}, tag='Training', phase=phase, add_global_step=True)

            # Optional image logging (unchanged)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                    self.base_diffusion._scale_input(z_t, tt),
                    self.autoencoder,
                )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
               
                ssim_str = f"_SSIM{loss.get('ssim', torch.tensor(0.0)).mean().item():.4f}"
                lpips_str = f"_LPIPS{loss.get('lpips', torch.tensor(0.0)).mean().item():.4f}"
                mse_str   = f"_MSE{loss.get('mse', torch.tensor(0.0)).mean().item():.4f}"
                tv_str    = f"_TV{loss.get('tv', torch.tensor(0.0)).mean().item():.5f}" if 'tv' in loss else ""

                custom_suffix = f"_iter{self.current_iters}{ssim_str}{lpips_str}{mse_str}{tv_str}"

                self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, add_global_step=True, custom_suffix=custom_suffix)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elapsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elapsed:.2f}s")
                self.logger.info("=" * 100)


def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == "__main__":
    import importlib
    from omegaconf import OmegaConf

    config_path = "./configs/realsr_swinunet_realesrgan256_journal.yaml"
    configs = OmegaConf.load(config_path)

    # Extract the trainer class path from the config
    target = configs.trainer.target  # e.g., 'trainer.TrainerDifIRLPIPS'
    module_name, class_name = target.rsplit(".", 1)

    # Dynamically import the module and class
    module = importlib.import_module(module_name)
    TrainerClass = getattr(module, class_name)

    # Instantiate trainer
    trainer = TrainerClass(configs)

    # Allow validation-only mode via config
    if configs.get("validate_only", False):
        trainer.init_logger()
        trainer.build_model()
        trainer.resume_from_ckpt()
        trainer.build_dataloader()
        trainer.validation()
    else:
        trainer.train()





# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
from torch_xla.distributed.parallel_loader import ParallelLoader
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from PIL import Image
from header import *
from rqvae.models.interfaces import Stage1ModelOutput, Stage2ModelOutput

logger = logging.getLogger(__name__)
DEBUG = bool(os.environ.get("DEBUG", 0))


class TrainerTemplate:
    def __init__(
        self,
        model,
        model_ema,
        dataset_trn,
        dataset_val,
        config,
        writer,
        device,
        distenv,
        model_aux=None,
        *,
        disc_state_dict=None,  # only used in VQGAN trainer
        eval:bool = False,
    ):
        super().__init__()

        num_workers = 16

        if DEBUG:
            if not dist.is_initialized():
                num_workers = 0
            config.experiment.test_freq = 1
            config.experiment.save_ckpt_freq = 1

        self.model = model
        self.model_ema = model_ema
        self.model_aux = model_aux
        self.dtype = self.model.module.get_last_layer().dtype
        self.config = config
        self.writer = writer
        self.device = device
        self.is_eval = eval
        self.distenv = distenv
        self.accu_step = config.experiment.accu_step
        self.actual_batch_size = config.experiment.actual_batch_size
        self.dataset_trn = dataset_trn
        self.dataset_val = dataset_val
        self.sampler_trn = torch.utils.data.distributed.DistributedSampler(
            self.dataset_trn,
            num_replicas=self.distenv.world_size,
            rank=self.distenv.world_rank,
            shuffle=True,
            seed=self.config.seed,
        )
        self.loader_trn = DataLoader(
            self.dataset_trn,
            sampler=self.sampler_trn,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
            drop_last=True, # very important for xla to avoid dynamic shape (sometimes)
            collate_fn=self.dataset_trn.collate_fn if hasattr(self.dataset_trn, 'collate_fn') else None
        )

        self.sampler_val = torch.utils.data.distributed.DistributedSampler(
            self.dataset_val,
            num_replicas=self.distenv.world_size,
            rank=self.distenv.world_rank,
            shuffle=False,
        )
        self.loader_val = DataLoader(
            self.dataset_val,
            sampler=self.sampler_val,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
            drop_last=False, # in validation, we need to make sure all data is used
            collate_fn=self.dataset_val.collate_fn if hasattr(self.dataset_val, 'collate_fn') else None
        )
        if self.distenv.master:
            logger.info(
                f"train dataset size: {len(dataset_trn)}, valid dataset size: {len(dataset_val)}"
            )
        if dist.get_world_size() > 1 and self.distenv.TPU:
            assert (
                dist.is_initialized()
            ), "Distributed training is not initialized when using multiple xla device"
            self.parallel_loader_trn = ParallelLoader(self.loader_trn, [self.device])
            self.parallel_loader_val = ParallelLoader(
                self.loader_val, [self.device]
            )  # in xla we use pl
        else:
            raise NotImplementedError
    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        raise NotImplementedError

    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        raise NotImplementedError
    def wrap_loader(self, split:str = 'train'):
        assert split in ['train', 'valid'], f"split should be either 'train' or 'valid', but got {split}"
        loader = self.loader_trn if split == 'train' else self.loader_val
        if self.distenv.TPU:
            if self.distenv.master:
                print(f"[!] TPU: using per_device_loader for validation")
            loader = ParallelLoader(loader, [self.device]).per_device_loader(self.device)
        else:
            if self.distenv.master:
                print(f"[!] NO TPU: using normal loader for validation")
        return loader
    @torch.no_grad()
    def batch_infer(self, valid:bool = True , save_root:str=None):
        assert os.path.exists(save_root), f"save_root {save_root} does not exist"
        model = self.model
        model.eval()
        loader = self.wrap_loader('valid' if valid else 'train')
        pbar = tqdm(enumerate(loader), desc='Inferencing', disable=not self.distenv.master,total=len(loader))
        for it, inputs in pbar:
            model.zero_grad()
            xs = inputs[0].to(self.device).to(self.dtype)
            img_paths = inputs[1]
            outputs:Stage1ModelOutput = model.module.infer(xs)
            xs_recon_or_gen = outputs.xs_recon
            xm.mark_step()
            for i, img_path in enumerate(img_paths):
                img_name = os.path.basename(img_path)
                save_path = os.path.join(save_root, img_name)
                img = xs_recon_or_gen[i].to(torch.float32).cpu().clamp(0, 1).numpy() 
                img = (img * 255).astype('uint8').transpose(1, 2, 0)
                img = Image.fromarray(img)
                img.save(save_path)
    def run_epoch(self, optimizer=None, scheduler=None, epoch_st=0):
        scaler = GradScaler() if self.config.experiment.amp else None
        for i in range(epoch_st, self.config.experiment.epochs):
            self.sampler_trn.set_epoch(i)
            if i % self.config.experiment.save_ckpt_freq == 0:
                self.save_ckpt(optimizer, scheduler, i + 1)
            summary_trn = self.train(optimizer, scheduler, scaler, epoch=i)
            xm.mark_step()
            if i == 0 or (i + 1) % self.config.experiment.test_freq == 0:
                summary_val = self.eval(epoch=i)
                if self.model_ema is not None:
                    summary_val_ema = self.eval(ema=True, epoch=i)
            if self.distenv.master:
                self.logging(
                    summary_trn, scheduler=scheduler, epoch=i + 1, mode="train"
                )
                if i == 0 or (i + 1) % self.config.experiment.test_freq == 0:
                    self.logging(
                        summary_val, scheduler=scheduler, epoch=i + 1, mode="valid"
                    )
                    if self.model_ema is not None:
                        self.logging(
                            summary_val_ema,
                            scheduler=scheduler,
                            epoch=i + 1,
                            mode="valid_ema",
                        )
            xm.rendezvous(
                "epoch_sync"
            )  # make sure we save the model properly without stuck
        self.save_ckpt(optimizer, scheduler, -1 )# last ckpt
    def _load_model_only(self, load_path, additional_attr_to_load:tuple = (), load_from_master:bool = True):
        global CKPT_FOLDER, MODEL_NAME, EMA_MODEL_NAME, ADDIONTIONAL_NAME
        rank = self.distenv.local_rank if not load_from_master else 0 # load from master (rank:0)
        model_path = os.path.join(load_path, MODEL_NAME.format(rank))
        model_weight = torch.load(model_path) # to xla directly
        self.model.module.load_state_dict(model_weight)
        if self.model_ema:
            ema_model_path = os.path.join(load_path, EMA_MODEL_NAME.format(rank))
            if os.path.exists(ema_model_path):
                ema_model_weight = torch.load(ema_model_path)
                self.model_ema.module.load_state_dict(ema_model_weight)
            else:
                xm.master_print(f"[!] EMA model path {ema_model_path} does not exist, skip loading EMA model")
        additional_path = os.path.join(load_path, ADDIONTIONAL_NAME.format(rank))
        additional_attr_ckpt = torch.load(additional_path)
        for attr in additional_attr_to_load:
            assert attr in additional_attr_ckpt, f"additional_attr_to_load {attr} not in additional_attr_ckpt"
            assert hasattr(self, attr), f"self does not have attribute {attr}"
            getattr(self, attr).module.load_state_dict(additional_attr_ckpt[attr])
    def _load_ckpt(self,load_path, optimizer, scheduler, additional_attr_to_load:tuple = (), load_from_master:bool = True):
        global CKPT_FOLDER, MODEL_NAME, OPT_NAME, SCH_NAME, ADDIONTIONAL_NAME, EMA_MODEL_NAME
        rank = self.distenv.local_rank if not load_from_master else 0
        ckpt_folder = load_path
        model_path = os.path.join(ckpt_folder, MODEL_NAME.format(rank))
        opt_path = os.path.join(ckpt_folder, OPT_NAME.format(rank))
        sch_path = os.path.join(ckpt_folder, SCH_NAME.format(rank))
        additional_path = os.path.join(ckpt_folder, ADDIONTIONAL_NAME.format(rank))
        model_weight = torch.load(model_path)
        optimizer_weight = torch.load(opt_path)
        scheduler_weight = torch.load(sch_path)
        additional_attr_ckpt = torch.load(additional_path)
        self.model.module.load_state_dict(model_weight)
        optimizer.load_state_dict(optimizer_weight)
        scheduler.load_state_dict(scheduler_weight)
        for attr in additional_attr_to_load:
            assert attr in additional_attr_ckpt, f"additional_attr_to_load {attr} not in additional_attr_ckpt"
            assert hasattr(self, attr), f"self does not have attribute {attr}"
            getattr(self, attr).module.load_state_dict(additional_attr_ckpt[attr])
        if self.model_ema:
            ema_model_path = os.path.join(ckpt_folder, EMA_MODEL_NAME.format(rank))
            if os.path.exists(ema_model_path):
                ema_model_weight = torch.load(ema_model_path)
                self.model_ema.module.load_state_dict(ema_model_weight)
            else:
                xm.master_print(f"[!] EMA model path {ema_model_path} does not exist, skip loading EMA model")
    def sync_and_to_cpu(self, state_dict, key_match:Optional[dict] = None):
        """
        """
        def convert_fn(item):
            if isinstance(item, torch.Tensor):
                item = xm._maybe_convert_to_cpu(item).to(torch.float32) # to cpu and float32
                return item
            elif isinstance(item, dict):
                return {k: convert_fn(v) for k,v in item.items()}
            elif isinstance(item, list):
                return [convert_fn(v) for v in item]
            elif isinstance(item, tuple):
                return tuple(convert_fn(v) for v in item)
            else:
                return item
        state_dict = {
            k: convert_fn(v) for k,v in state_dict.items() if key_match is None or k in key_match
        }
        return state_dict
    def save_ckpt(self, optimizer, scheduler, epoch, additional_attr_to_save:tuple = (), master_only:bool = True):
        global CKPT_FOLDER, MODEL_NAME, OPT_NAME, SCH_NAME, ADDIONTIONAL_NAME, EMA_MODEL_NAME
        if master_only and not self.distenv.master:
            return
        epoch = 'last' if epoch == -1 else epoch
        rank = self.distenv.local_rank
        ckpt_folder = os.path.join(self.config.result_path , CKPT_FOLDER.format(epoch))
        model_path = os.path.join(ckpt_folder, MODEL_NAME.format(rank))
        opt_path = os.path.join(ckpt_folder, OPT_NAME.format(rank))
        sch_path = os.path.join(ckpt_folder, SCH_NAME.format(rank))
        additional_path = os.path.join(ckpt_folder, ADDIONTIONAL_NAME.format(rank))
        os.makedirs(ckpt_folder, exist_ok=True)
        model_weight = self.sync_and_to_cpu(self.model.module.state_dict())
        optimizer_weight = self.sync_and_to_cpu(optimizer.state_dict())
        scheduler_weight = self.sync_and_to_cpu(scheduler.state_dict())
        additional_attr_ckpt = {}
        for attr in additional_attr_to_save:
            additional_attr_ckpt[attr] = self.sync_and_to_cpu(getattr(self, attr).module.state_dict())
        torch.save(model_weight, model_path)
        torch.save(optimizer_weight, opt_path)
        torch.save(scheduler_weight, sch_path)
        if len(additional_attr_ckpt) > 0:
            torch.save(additional_attr_ckpt, additional_path)
        if self.model_ema:
            ema_model_path = os.path.join(ckpt_folder, EMA_MODEL_NAME.format(rank))
            ema_model_weight = self.model_ema.state_dict()
            torch.save(ema_model_weight, ema_model_path)
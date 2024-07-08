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

import logging
import os

import torch
import torchvision
from tqdm import tqdm

import rqvae.utils.dist as dist_utils

from .accumulator import AccmStage1WithGAN
from .trainer import TrainerTemplate
import torch_xla.core.xla_model as xm
logger = logging.getLogger(__name__)
import os
DEBUG = bool(os.environ.get("DEBUG", 0))
import time # for debugging

class Trainer(TrainerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_accm(self):
        metric_names = [
            'loss_total', 
        ]
        accm = AccmStage1WithGAN(
            metric_names,
            device=self.device,
        )

        return accm

    @torch.no_grad()
    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.wrap_loader('valid' if valid else 'train')
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()

        pbar = tqdm(enumerate(loader), total=len(loader),disable= not self.distenv.master)
        model.eval()
        for it, inputs in pbar:
            model.zero_grad()
            xs = inputs[0].to(self.device)
            stage_1_output, stage_2_output = model(xs)
            zs = stage_1_output[0]
            zs_pred = stage_2_output[0]
            outputs = model.module.compute_loss(zs_pred, zs, *stage_1_output[1:], *stage_2_output[1:], xs=xs, valid=True)
            xm.mark_step()
            loss_gen = outputs['loss_total'] # logging
            loss_total = loss_gen
            metrics = dict(loss_total=loss_total)
            accm.update(metrics,
                        count=1,
                        sync=True,
                        distenv=self.distenv)
            line = accm.get_summary().print_line()
            pbar.set_description(line)
        line = accm.get_summary(n_inst).print_line() 
        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            self.reconstruct(zs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary['zs'] = zs

        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        model.train()
        model.zero_grad(set_to_none=True)


        accm = self.get_accm()
        loader = self.wrap_loader('train')
        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)
        if DEBUG:
            xm.mark_step()
            it_st_time = time.time()
            xm.master_print(f"[!]start time: {it_st_time}s")
            
        for it, inputs in pbar:
            xs = inputs[0].to(self.device, non_blocking=True)
            stage_1_output, stage_2_output = model(xs)
            zs = stage_1_output[0]
            zs_pred = stage_2_output[0]
            outputs = model.module.compute_loss(zs_pred, zs, *stage_1_output[1:], *stage_2_output[1:], xs=xs)
            xm.mark_step()
            loss_gen = outputs['loss_total']
            loss_gen_total = loss_gen 
            loss_gen_total.backward()
            if (it + 1) % self.accu_step == 0:
                optimizer.step() # in DDP we use optimizer.step() instead of xm.optimizer_step(optimizer)
                scheduler.step()
                model.zero_grad(set_to_none=True)
            xm.mark_step()
            # logging
            loss_total = loss_gen_total.detach() 
            metrics = {
                'loss_total': loss_total,
            }
            accm.update(metrics, count=1)
            if it == 2 and DEBUG:
                xm.mark_step()
                en_compile_time = time.time()
                xm.master_print(f"[!]compile time: {en_compile_time - it_st_time}s")
                # make sure every process is in sync
                exit()
            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)
                # per-step logging
                global_iter = epoch * len(self.loader_trn) + it
                if (global_iter+1) % 50 == 0:
                    for key, value in metrics.items():
                        if isinstance(value, torch.Tensor):
                            value = value.to(torch.float32) # bf16 does not support directly conversion to numpy yet
                        self.writer.add_scalar(f'loss_step/{key}', value, 'train', global_iter)
                    self.writer.add_scalar('lr_step', scheduler.get_last_lr()[0], 'train', global_iter)
                if (global_iter+1) % 250 == 0:
                    bsz = xs.size(0)
                    if bsz == 1: # need to be handle properly
                        continue
                    max_shard_size = min(bsz,16)
                    zs, zs_pred = model.module.get_recon_imgs(zs[:max_shard_size], zs_pred[:max_shard_size])
                    print(zs.shape, zs_pred.shape)
                    grid = torch.cat([zs[:max_shard_size//2], zs_pred[:max_shard_size//2], zs[max_shard_size//2:], zs_pred[max_shard_size//2:]], dim=0)
                    grid = torchvision.utils.make_grid(grid, nrow=max_shard_size//2).detach().cpu().float()
                    self.writer.add_image('reconstruction_step', grid, 'train', global_iter)
        summary = accm.get_summary()
        summary['zs'] = zs

        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode='train'):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_freq == 0:
            self.reconstruct(summary['zs'], epoch, mode)
        for key, value in summary.metrics.items():
            self.writer.add_scalar(f'loss/{key}', summary[key], mode, epoch)
        if mode == 'train':
            self.writer.add_scalar('lr', scheduler.get_last_lr()[0], mode, epoch)
        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""
        logger.info(line)

    @torch.no_grad()
    def reconstruct(self, zs, epoch, mode='valid'):
        # do not write image when bs is 1
        if zs.size(0) == 1:
            return
        bsz = zs.size(0)
        max_shard_size = min((bsz//2)*2, 16)
        model = self.model_ema if 'ema' in mode else self.model
        model.eval()
        zs_real = zs[:max_shard_size]
        zs_recon = model.module.stage_2_model(zs_real)[0]
        zs_real, zs_recon = model.module.get_recon_imgs(zs_real, zs_recon)
        grid = torch.cat([zs_real[:max_shard_size//2], zs_recon[:max_shard_size//2], zs_real[max_shard_size//2:], zs_recon[max_shard_size//2:]], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=8).detach().cpu().float()
        self.writer.add_image('reconstruction', grid, mode, epoch)
    def _load_ckpt(self, optimizer, scheduler, epoch: int = -1, load_from_master=True):
        return super()._load_ckpt(optimizer, scheduler, epoch,load_from_master=load_from_master)
    def _load_model_only(self, load_path, load_from_master=True):
        return super()._load_model_only(load_path, load_from_master=load_from_master)
    def save_ckpt(self, optimizer, scheduler, epoch):
        return super().save_ckpt(optimizer, scheduler, epoch)
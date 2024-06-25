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
            outputs = model(xs)
            xs_gen = outputs[0]
            outputs = model.module.compute_loss(*outputs, xs=xs, valid=True)
            xm.mark_step()
            loss_gen = outputs['loss_total']
            logits = {k: v * xs.size(0) for k, v in logits.items()}
            # logging
            loss_total = loss_gen
            metrics = dict(loss_total=loss_total,
                            **logits,
                        )
            accm.update(metrics,
                        count=xs.shape[0],
                        sync=True,
                        distenv=self.distenv)
            line = accm.get_summary().print_line() # moving this into the below if block would cause forever hanging... don't know why
            if self.distenv.master:
                pbar.set_description(line)
        line = accm.get_summary(n_inst).print_line() 
        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            self.reconstruct(xs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary['xs'] = xs

        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        model.train()

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
            model.zero_grad(set_to_none=True)
            xs = inputs[0].to(self.device, non_blocking=True)
            outputs = model(xs)
            xs_recon = outputs[0]
            outputs = model.module.compute_loss(*outputs, xs=xs)
            xm.mark_step()
            loss_gen = outputs['loss_total']
            loss_gen_total = loss_gen 
            loss_gen_total.backward()
            optimizer.step() # in DDP we use optimizer.step() instead of xm.optimizer_step(optimizer)
            scheduler.step()
            xm.mark_step()
            xm.mark_step()
            # logging
            loss_total = loss_gen_total.detach() 
            metrics = {
                'loss_total': loss_total,
                'loss_gen': loss_gen.detach(),
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
                    xs_real, xs_recon = model.module.get_recon_imgs(xs[:16], xs_recon[:16])
                    grid = torch.cat([xs_real[:8], xs_recon[:8], xs_real[8:], xs_recon[8:]], dim=0)
                    grid = torchvision.utils.make_grid(grid, nrow=8)
                    self.writer.add_image('reconstruction_step', grid, 'train', global_iter)
        summary = accm.get_summary()
        summary['xs'] = xs

        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode='train'):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_freq == 0:
            self.reconstruct(summary['xs'], epoch, mode)
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
    def reconstruct(self, xs, epoch, mode='valid'):
        model = self.model_ema if 'ema' in mode else self.model
        model.eval()

        xs_real = xs[:16]
        xs_recon = model(xs_real)[0]
        xs_real, xs_recon = model.module.get_recon_imgs(xs_real, xs_recon)

        grid = torch.cat([xs_real[:8], xs_recon[:8], xs_real[8:], xs_recon[8:]], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=8)
        self.writer.add_image('reconstruction', grid, mode, epoch)


    def save_ckpt(self, optimizer, scheduler, epoch):
        ckpt_path = os.path.join(self.config.result_path, 'epoch%d_model.pt' % epoch)
        logger.info("epoch: %d, saving %s", epoch, ckpt_path)
        ckpt = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if self.model_ema is not None:
            ckpt.update(state_dict_ema=self.model_ema.module.module.state_dict())
        torch.save(ckpt, ckpt_path)

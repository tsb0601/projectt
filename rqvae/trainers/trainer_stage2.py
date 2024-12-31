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
import time
import torch
import torchvision
from tqdm import tqdm

import rqvae.utils.dist as dist_utils

from .accumulator import AccmStage1WithGAN, SummaryStage1WithGAN
from .trainer import TrainerTemplate
from header import *
import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast
from contextlib import nullcontext
logger = logging.getLogger(__name__)
import os

DEBUG = bool(os.environ.get("DEBUG", 0))
from typing import *
from rqvae.models.interfaces import (
    Stage1ModelOutput,
    Stage2ModelOutput,
    Stage1Encodings,
    Stage2ModelWrapper,
)
from rqvae.img_datasets.interfaces import LabeledImageData

class Trainer(TrainerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_accm(self):
        metric_names = [
            "loss_total",
            "loss_total_ema",
            "grad_norm",
        ]
        accm = AccmStage1WithGAN(
            metric_names,
            device=self.device,
        )

        return accm

    @torch.no_grad()
    def eval(
        self, valid=True, ema=False, verbose=False, epoch=0
    ) -> SummaryStage1WithGAN:
        model = self.model_ema if ema else self.model
        loader = self.wrap_loader("valid" if valid else "train")
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()

        pbar = tqdm(
            enumerate(loader), total=len(loader), disable=not self.distenv.master
        )
        model.eval()
        last_input = None
        for it, inputs in pbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            last_input = inputs
            with autocast(self.device) if self.use_autocast else nullcontext():
                stage1_encodings, stage2_output = model(inputs)
                stage1_encodings: Stage1Encodings
                stage2_output: Stage2ModelOutput
                zs = stage1_encodings.zs
                outputs = self.model_woddp.compute_loss(
                    stage1_encodings, stage2_output, inputs, valid=True
                )
                xm.mark_step()
                loss_gen = outputs["loss_total"]  # logging
            loss_total = loss_gen
            metrics = dict(loss_total=loss_total)
            accm.update(metrics, count=1, sync=True, distenv=self.distenv)
            line = f"""(epoch {epoch} / iter {it}) """
            line += accm.get_summary().print_line()
            #for metric_name, value in metrics.items():
            #    line += f""", {metric_name}: {value:4f}"""
            pbar.set_description(line)
        line = accm.get_summary().print_line()
        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            with torch.no_grad():
                self.generate(last_input, epoch, mode)
        summary = accm.get_summary()
        summary["input"] = last_input

        return summary

    def train(
        self, optimizer=None, scheduler=None, scaler=None, epoch=0
    ) -> SummaryStage1WithGAN:
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        accm = self.get_accm()
        loader = self.wrap_loader("train")
        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)
        last_input = None
        for it, inputs in pbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            last_input = inputs
            xs = inputs.img
            with autocast(self.device) if self.use_autocast else nullcontext():
                stage1_encodings, stage2_output = self.model(inputs)
                stage1_encodings: Stage1Encodings
                stage2_output: Stage2ModelOutput
                outputs = self.model_woddp.compute_loss(stage1_encodings, stage2_output, inputs)
                loss = outputs["loss_total"] # always use float for loss
            loss.backward()
            if self.clip_grad_norm > 0:
                grad_norm = self.norm_tracker.clip_norm()
                #after_clip_grad_norm = self.norm_tracker()
                #xm.master_print(f"after_clip_grad_norm: {after_clip_grad_norm}")
            else:
                grad_norm = torch.tensor(-1) # not tracking
            xm.mark_step()
            # logging
            loss_total = loss.detach()
            metrics = {
                "loss_total": loss_total,
                "grad_norm": grad_norm.detach(),
            }
            if (it + 1) % self.accu_step == 0:
                if self.use_ddp:
                    optimizer.step()  # in DDP we use optimizer.step() instead of xm.optimizer_step(optimizer), see https://github.com/pytorch/xla/blob/master/docs/ddp.md for performance tips
                else:
                    xm.optimizer_step(optimizer) # else we use xm.optimizer_step
                self.model.zero_grad(set_to_none=True)
                scheduler.step()
                if self.model_ema_woddp is not None:
                    self.model_ema_woddp.update(self.model_woddp, step=None) # use fixed decay
            xm.mark_step()
            accm.update(metrics, count=1, sync=True, distenv=self.distenv) # in training we only monitor master process for logging
            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)
                # per-step logging
                global_iter = epoch * len(self.loader_trn) + it
                if (global_iter + 1) % (20 * self.accu_step) == 0: # log every 20 actual steps
                    for key, value in metrics.items():
                        if isinstance(value, torch.Tensor):
                            value = value.to(
                                torch.float32
                            )  # bf16 does not support directly conversion to numpy yet
                        self.writer.add_scalar(
                            f"loss_step/{key}", value, "train", global_iter
                        )
                    self.writer.add_scalar(
                        "lr_step", scheduler.get_last_lr()[0], "train", global_iter
                    )
                    if self.model_ema is not None: # if ema exist we also track an ema loss
                        with torch.no_grad():
                            with autocast(self.device) if self.use_autocast else nullcontext():
                                stage1_encodings, stage2_output = self.model_ema(inputs)
                                outputs = self.model_ema_woddp.compute_loss(stage1_encodings, stage2_output, inputs)
                                loss_total_ema = outputs["loss_total"]
                        self.writer.add_scalar(
                            "loss_step/loss_total_ema", loss_total_ema, "train", global_iter
                        )
                if it % (len(loader) // 2) == 0:
                    self.model.eval()
                    with torch.no_grad():
                        bsz = xs.size(0)
                        bsz = len(inputs)
                        max_shard_size = min(bsz, 16)
                        inputs = inputs[:max_shard_size]
                        self.model_woddp: Stage2ModelWrapper
                        with autocast(self.device) if self.use_autocast else nullcontext():
                            infer_output = self.model_woddp.infer(inputs)
                        xs = infer_output.xs_recon.clamp(0, 1)
                        grid = torchvision.utils.make_grid(xs, nrow=4).detach().cpu().float()
                        self.writer.add_image(
                            "generation_step", grid, "train", global_iter
                        )
                    self.model.train()
        
        summary = accm.get_summary()
        summary["input"] =  last_input

        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_freq == 0:
            self.generate(summary["input"], epoch, mode)
        for key, value in summary.metrics.items():
            self.writer.add_scalar(f"loss/{key}", summary[key], mode, epoch)
        if mode == "train":
            self.writer.add_scalar("lr", scheduler.get_last_lr()[0], mode, epoch)
        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""
        logger.info(line)
    @torch.no_grad()
    def generate(self, inputs:LabeledImageData, epoch:int, mode="valid"):
        self.model.eval()
        bsz = len(inputs)
        max_shard_size = min(bsz, 16)
        inputs = inputs[:max_shard_size]
        inputs._to(self.device)._to(self.dtype)
        self.model_woddp: Stage2ModelWrapper
        infer_output = self.model_woddp.infer(inputs)
        xs = infer_output.xs_recon.clamp(0, 1)
        grid = torchvision.utils.make_grid(xs, nrow=4).detach().cpu().float()
        self.writer.add_image("generation", grid, mode, epoch)
    @torch.no_grad()
    def reconstruct(self, zs, epoch, mode="valid"):
        raise NotImplementedError("reconstruct method is not implemented in the stage2 trainer")

    def _load_ckpt(self, optimizer, scheduler, epoch: int = -1, load_from_master=True,additional_attr_to_load = ()):
        return super()._load_ckpt(
            optimizer, scheduler, epoch, load_from_master=load_from_master, additional_attr_to_load=additional_attr_to_load
        )

    def _load_model_only(self, load_path, load_from_master=True, additional_attr_to_load = ()):
        return super()._load_model_only(load_path, load_from_master=load_from_master, additional_attr_to_load=additional_attr_to_load)

    def save_ckpt(self, optimizer, scheduler, epoch):
        return super().save_ckpt(optimizer, scheduler, epoch)
    
    def _save_model_only(self, epoch, additional_attr_to_save: tuple = ()):
        return super()._save_model_only(epoch, additional_attr_to_save = additional_attr_to_save)

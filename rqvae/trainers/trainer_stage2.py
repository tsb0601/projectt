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

from cv2 import merge
from sentry_sdk import last_event_id
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
from matplotlib import pyplot as plt
logger = logging.getLogger(__name__)
import os

DEBUG = bool(os.environ.get("DEBUG", 0))
import time  # for debugging
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
        track_x = []
        track_y = []
        last_input = None
        for it, inputs in pbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            last_input = inputs
            xs = inputs.img
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
                x, y = outputs["valid"] # t, loss by default
                track_x.append(x)
                track_y.append(y)
            loss_total = loss_gen
            metrics = dict(loss_total=loss_total)
            accm.update(metrics, count=1, sync=True, distenv=self.distenv)
            line = accm.get_summary().print_line()
            pbar.set_description(line)
        line = accm.get_summary(n_inst).print_line()
        track_x = torch.cat(track_x, 0)
        track_y = torch.cat(track_y, 0)
        all_x = xm.mesh_reduce("all_x", track_x, torch.cat)
        all_y = xm.mesh_reduce("all_y", track_y, torch.cat)
        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            with torch.no_grad():
                self.generate(last_input, epoch, mode)
            # logging x, y 
            # try merge y w same x
            
            all_x = all_x.cpu().numpy()
            all_y = all_y.float().cpu().numpy()
            merged = {}
            for x, y in zip(all_x, all_y):
                if x not in merged:
                    merged[x] = []
                merged[x].append(y)
            all_x = list(merged.keys())
            all_x.sort()
            all_y = [sum(merged[x]) / len(merged[x]) for x in all_x]
            import numpy as np
            npy_x = np.array(all_x)
            npy_y = np.array(all_y)
            torch.save({
                "x": npy_x,
                "y": npy_y
            }, f"plot_{mode}.pt")
            plt.plot(all_x, all_y, label=f"{mode}")
            plt.legend()
            plt.savefig(f"plot_{mode}.png")
        summary = accm.get_summary(n_inst)
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
        if DEBUG:
            xm.mark_step()
            it_st_time = time.time()
            xm.master_print(f"[!]start time: {it_st_time}s")
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
                zs = stage1_encodings.zs
                zs_pred = stage2_output.zs_pred
                outputs = self.model_woddp.compute_loss(stage1_encodings, stage2_output, inputs)
                xm.mark_step()
                loss = outputs["loss_total"]
            loss.backward()
            if (it + 1) % self.accu_step == 0:
                if self.use_ddp:
                    optimizer.step()  # in DDP we use optimizer.step() instead of xm.optimizer_step(optimizer), see https://github.com/pytorch/xla/blob/master/docs/ddp.md for performance tips
                else:
                    xm.optimizer_step(optimizer) # else we use xm.optimizer_step
                scheduler.step()
                self.model.zero_grad(set_to_none=True)
            xm.mark_step()
            # logging
            loss_total = loss.detach()
            metrics = {
                "loss_total": loss_total,
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
                if (global_iter + 1) % 20 == 0:
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
                if (global_iter + 1) % 500 == 0:
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
            xm.mark_step() # wait for main process to finish logging
        
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

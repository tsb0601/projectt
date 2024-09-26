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
import trace

import torch
import torchvision
from tqdm import tqdm

from rqvae.losses.vqgan import (
    create_vqgan_loss,
    create_discriminator_with_optimizer_scheduler,
)
import rqvae.utils.dist as dist_utils

from .accumulator import AccmStage1WithGAN, SummaryStage1WithGAN
from .trainer import TrainerTemplate
from header import *
import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast
from contextlib import nullcontext
logger = logging.getLogger(__name__)
import os
import numpy as np
DEBUG = bool(os.environ.get("DEBUG", 0))
import time  # for debugging
from typing import *
from rqvae.models.interfaces import *
from rqvae.img_datasets.interfaces import LabeledImageData

def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


class Trainer(TrainerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # GAN related part
        gan_config = self.config.gan

        disc_config = gan_config.disc
        self.gan_start_epoch = gan_config.loss.disc_start
        num_epochs_for_gan = self.config.experiment.epochs - self.gan_start_epoch
        self.lpips_start_epoch = gan_config.loss.lpips_start
        disc_model, disc_optim, disc_sched = (
            create_discriminator_with_optimizer_scheduler(
                disc_config,
                steps_per_epoch=len(self.loader_trn),
                max_epoch=num_epochs_for_gan,
                device = self.device,
                dtype = self.dtype,
                distenv=self.distenv,
                is_eval=self.is_eval,
            )
        )
        disc_state_dict = kwargs.get("disc_state_dict", None)
        if disc_state_dict is not None:
            disc_model.load_state_dict(disc_state_dict)
            logger.info("[state] discriminator loaded")

        self.discriminator = dist_utils.dataparallel_and_sync(self.distenv, disc_model)
        self.disc_optimizer = disc_optim
        self.disc_scheduler = disc_sched

        d_loss, g_loss, p_loss = create_vqgan_loss(gan_config.loss)

        self.disc_loss = d_loss
        self.gen_loss = g_loss
        self.perceptual_loss = p_loss.to(self.device).to(self.dtype).eval()
        self.perceptual_weight = gan_config.loss.perceptual_weight
        self.disc_weight = gan_config.loss.disc_weight
        self.get_last_layer = self.model_woddp.get_last_layer
    def get_accm(self):
        metric_names = [
            "loss_total",
            "loss_recon",
            "loss_latent",
            "loss_pcpt",
            "loss_gen",
            "loss_disc",
            "g_weight",
            "logits_real",
            "logits_fake",
            "grad_norm"
        ]
        accm = AccmStage1WithGAN(
            metric_names,
            device=self.device,
        )

        return accm

    def gan_loss(self, inputs, recons, mode="idle"):

        loss_gen = torch.zeros((), device=self.device)
        loss_disc = torch.zeros((), device=self.device)

        logits_avg = {}

        if mode == "gen":
            logits_fake, _ = self.discriminator(recons.contiguous(), None)
            loss_gen = self.gen_loss(logits_fake)

        elif mode == "disc":
            logits_fake, logits_real = self.discriminator(
                recons.contiguous().detach(), inputs.contiguous().detach()
            )

            loss_disc = self.disc_loss(logits_real, logits_fake)

            logits_avg["logits_real"] = logits_real.detach().mean()
            logits_avg["logits_fake"] = logits_fake.detach().mean()

        elif mode == "eval":
            logits_fake, logits_real = self.discriminator(
                recons.contiguous().detach(), inputs.contiguous().detach()
            )

            loss_gen = self.gen_loss(logits_fake)
            loss_disc = self.disc_loss(logits_real, logits_fake)

            logits_avg["logits_real"] = logits_real.detach().mean()
            logits_avg["logits_fake"] = logits_fake.detach().mean()

        return loss_gen, loss_disc, logits_avg
    @torch.no_grad()
    def encode_only(self, valid:bool= True):
        """
        This func is used to calculate the mean and variance of the latent space
        The model should return a running mean and variance of the latent space
        """
        self.model.train() # to calculate the running mean and variance
        loader = self.wrap_loader("valid" if valid else "train")
        for it, inputs in tqdm(enumerate(loader)):
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            with autocast(self.device) if self.use_autocast else nullcontext():
                stage1_encodings: Stage1Encodings = self.model_woddp.encode(inputs)
                running_mean = stage1_encodings.additional_attr['running_mean']
                running_var = stage1_encodings.additional_attr['running_var']
            xm.mark_step()
        # save the running mean and variance
        return running_mean, running_var
    @torch.no_grad()
    def calculate_mean_and_std(self, valid:bool= True) -> nn.modules.batchnorm._BatchNorm:
        """
        This func is used to calculate the mean and variance of the latent space
        """
        self.model.eval()
        loader = self.wrap_loader("valid" if valid else "train")
        def get_batchnorm(latent_size: tuple)-> nn.modules.batchnorm._BatchNorm:
            bn_list = [nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            return bn_list[len(latent_size) - 2](latent_size[1], affine=False, track_running_stats=True)
        bn = None
        self.model_woddp: Stage1ModelWrapper
        if self.model_woddp.connector.bn is None:
            # do a test forward pass to determine the shape of the latent space
            test_input:LabeledImageData = next(iter(loader))
            test_input._to(self.device)._to(self.dtype)
            test_output = self.model_woddp.encode(test_input) # after encoder and connector
            bn = get_batchnorm(test_output.zs.shape).to(self.device).to(self.dtype)
            self.model_woddp.connector.bn = bn # hook a proper bn
            self.model_woddp.connector.bn.train()
            self.model_woddp.connector.hook_forward()
        tbar = tqdm(enumerate(loader), total=len(loader), disable=not self.distenv.master)
        for it, inputs in tbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            with autocast(self.device) if self.use_autocast else nullcontext():
                self.model_woddp.encode(inputs)
            xm.mark_step()
        # do sync for bn
        bn = self.model_woddp.connector.bn
        bn.eval()
        E_x = bn.running_mean
        Var_x = bn.running_var
        # all gather to get real mean and variance
        mean_per_device = E_x.unsqueeze(0).to(self.device) # 1, C
        mean_all = xm.all_gather(mean_per_device) # (N, C)
        mean_of_mean = mean_all.mean(dim=0) # C
        var_of_mean = mean_all.var(dim=0) # C
        var_per_device = Var_x.unsqueeze(0).to(self.device)
        var_all = xm.all_gather(var_per_device)
        mean_of_var = var_all.mean(dim=0)
        estimated_mean = mean_of_mean 
        estimated_var = mean_of_var + var_of_mean # var(reduced(x)) = E[var(x)] + var(E[x])
        xm.rendezvous('mean_var_sync')
        # update the bn
        self.model_woddp.connector.bn.running_mean = estimated_mean
        self.model_woddp.connector.bn.running_var = estimated_var
        self.model_woddp.connector.bn.eval()
        # do some test, uncomment this to test the correctness of the mean and variance
        """
        loader = self.wrap_loader("valid" if valid else "train")
        avg_mean = 0
        avg_var = 0
        avg_mse = 0
        for it, inputs in enumerate(loader):
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            with autocast(self.device) if self.use_autocast else nullcontext():
                stage1_encodings: Stage1Encodings = self.model_woddp.encode(inputs)
                zs = stage1_encodings.zs
                normed_ = self.model_woddp.connector.normalize(stage1_encodings)
                normed_zs = normed_.zs
                unnormed_ = self.model_woddp.connector.unnormalize(normed_)
                unnormed_zs = unnormed_.zs
                mse_l = torch.nn.functional.mse_loss(zs, unnormed_zs)
                avg_mean += normed_zs.mean()
                avg_var += normed_zs.var()
                avg_mse += mse_l
                #xm.master_print(f'[!]mean: {normed_zs.mean()}, var: {normed_zs.var()}, mse: {mse_l.item()}')
            xm.mark_step()
        avg_mean /= len(loader)
        avg_var /= len(loader)
        avg_mse /= len(loader)
        xm.master_print(f'[!]avg mean: {avg_mean}, avg var: {avg_var}, avg mse: {avg_mse}')
        """
        return self.model_woddp.connector.bn
    @torch.no_grad()
    def cache_latent(self, feature_path:str, valid:bool= True):
        self.model.eval()
        loader = self.wrap_loader("valid" if valid else "train")
        for it, inputs in tqdm(enumerate(loader)):
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            img_paths = inputs.img_path
            with autocast(self.device) if self.use_autocast else nullcontext():
                stage1_encodings: Stage1Encodings = self.model_woddp.encode(inputs)
                zs = stage1_encodings.zs
                label = inputs.condition
                for i, z in enumerate(zs):
                    img_name = os.path.basename(img_paths[i])
                    img_name = img_name.split('.')[0]
                    save_path = f'{feature_path}/{img_name}.npy'   
                    z_i = z.cpu().numpy()
                    label_i = label[i].cpu().numpy()
                    xm.master_print(f'[!]saving latent to {save_path}, latent shape: {z_i.shape}, label shape: {label_i.shape}')
                    xm.mark_step()
                    exit()
                    np.save(save_path, {
                        'z': z_i,
                        'label': label_i
                    })
            xm.mark_step()
    @torch.no_grad()
    def eval(
        self, valid:bool=True, ema:bool=False, verbose:bool=False, epoch:int=0
    ) -> SummaryStage1WithGAN:
        model = self.model_ema if ema else self.model
        discriminator = self.discriminator
        loader = self.wrap_loader("valid" if valid else "train")
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        use_discriminator = True if epoch >= self.gan_start_epoch else False

        accm = self.get_accm()

        pbar = tqdm(
            enumerate(loader), total=len(loader), disable=not self.distenv.master
        )
        model.eval()
        discriminator.eval()
        for it, inputs in pbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            xs = inputs.img
            with autocast(self.device) if self.use_autocast else nullcontext():
                stage1_output: Stage1ModelOutput = model(inputs)
                xs_recon = stage1_output.xs_recon  # calling convention
                outputs = self.model_woddp.compute_loss(stage1_output, inputs, valid=True)
                xm.mark_step()
                loss_rec_lat = outputs["loss_total"]
                loss_recon = outputs["loss_recon"]
                loss_latent = outputs["loss_latent"]

                loss_pcpt = (
                    self.perceptual_loss(xs, xs_recon)
                    if self.perceptual_weight > 0
                    else torch.zeros((), device=self.device, dtype=self.dtype)
                )
                p_weight = self.perceptual_weight

                if use_discriminator:
                    loss_gen, loss_disc, logits = self.gan_loss(xs, xs_recon, mode="eval")
                else:
                    loss_gen = torch.zeros((), device=self.device)
                    loss_disc = torch.zeros((), device=self.device)
                    logits = {}

            # logging
            loss_total = loss_rec_lat + p_weight * loss_pcpt  # rec + lat + pcpt
            metrics = dict(
                loss_total=loss_total,
                loss_recon=loss_recon,
                loss_latent=loss_latent,
                loss_pcpt=loss_pcpt,
                loss_gen=loss_gen,
                loss_disc=loss_disc,
                **logits,
            )
            accm.update(metrics, count=1, sync=True, distenv=self.distenv)
            line = (
                accm.get_summary().print_line()
            )  # moving this into master only would cause forever hanging... don't know why
            pbar.set_description(line)
        line = accm.get_summary(n_inst).print_line()
        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            self.reconstruct(xs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary["input"] = xs

        return summary

    def train(
        self, optimizer=None, scheduler=None, scaler=None, epoch=0
    ) -> SummaryStage1WithGAN:
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        discriminator = self.discriminator
        discriminator.train()
        discriminator.zero_grad(set_to_none=True)
        use_discriminator = True if epoch >= self.gan_start_epoch else False
        use_lpips = True if epoch >= self.lpips_start_epoch and self.perceptual_weight > 0 else False
        xm.master_print(f"[!]use_discriminator: {use_discriminator}")
        accm = self.get_accm()
        loader = self.wrap_loader("train")
        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)
        for it, inputs in pbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            xs = inputs.img
            with autocast(self.device) if self.use_autocast else nullcontext():
                stage1_output: Stage1ModelOutput = self.model(inputs)
                xs_recon = stage1_output.xs_recon  # calling convention
                outputs = self.model_woddp.compute_loss(stage1_output, inputs)
                loss_rec_lat = outputs["loss_total"]
                loss_recon = outputs["loss_recon"]
                loss_latent = outputs["loss_latent"]
                xm.mark_step()
                # generator loss
                loss_pcpt = (
                    self.perceptual_loss(xs, xs_recon)
                    if use_lpips
                    else torch.zeros((), device=self.device, dtype=self.dtype)
                )
                p_weight = self.perceptual_weight
            if use_discriminator:
                with autocast(self.device) if self.use_autocast else nullcontext():
                    loss_gen, _, _ = self.gan_loss(xs, xs_recon, mode="gen")
                g_weight = calculate_adaptive_weight( # calculate adaptive weight involves backward so it should be excluded from autocast
                    loss_recon + p_weight * loss_pcpt,
                    loss_gen,
                    last_layer=self.get_last_layer(),
                )
            else:
                loss_gen = torch.zeros((), device=self.device)
                g_weight = torch.zeros((), device=self.device)
            xm.mark_step()
            loss_gen_total = (
                loss_rec_lat
                + p_weight * loss_pcpt
                + g_weight * self.disc_weight * loss_gen
            )
            loss_gen_total.backward()
            if self.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            else:
                grad_norm = torch.tensor(-1) # not tracking
            if (it + 1) % self.accu_step == 0:
                if self.use_ddp:
                    optimizer.step()  # in DDP we use optimizer.step() instead of xm.optimizer_step(optimizer), see https://github.com/pytorch/xla/blob/master/docs/ddp.md for performance tips
                else:
                    xm.optimizer_step(optimizer) # else we use xm.optimizer_step
                self.model.zero_grad(set_to_none=True)
                scheduler.step()
                
            xm.mark_step()
            # discriminator loss

            if use_discriminator:
                with autocast(self.device) if self.use_autocast else nullcontext():
                    _, loss_disc, logits = self.gan_loss(xs, xs_recon, mode="disc")
                    dict_loss = loss_disc * self.disc_weight
                dict_loss.backward()
                # torch.autograd.backward(dict_loss, grad_tensors=[xs], retain_graph=False)
                if (it + 1) % self.accu_step == 0:
                    if self.use_ddp:
                        self.disc_optimizer.step()
                    else:
                        xm.optimizer_step(self.disc_optimizer)
                    self.disc_scheduler.step()
                    # discriminator.zero_grad(set_to_none=True)
                    self.model.zero_grad(set_to_none=True)
                    if self.model_ema_woddp is not None:
                        self.model_ema_woddp.update(self.model_woddp, step=None) # use fixed decay
            else:
                loss_disc = torch.zeros((), device=self.device)
                logits = {}
            xm.mark_step()
            # logging
            loss_total = (
                loss_rec_lat.detach() + p_weight * loss_pcpt.detach()
            )  # rec + lat + pcpt
            metrics = {
                "loss_total": loss_total,
                "loss_recon": loss_recon.detach(),
                "loss_latent": loss_latent.detach(),
                "loss_pcpt": loss_pcpt.detach(),
                "loss_gen": loss_gen.detach(),
                "loss_disc": loss_disc.detach(),
                "g_weight": g_weight.detach(),
                "grad_norm": grad_norm.detach(),
                **logits,
            }
            accm.update(metrics, count=1)
            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += (
                    accm.get_summary().print_line()
                )  # moving this to not master only cause deadlock
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                line += f""", d_lr: {self.disc_scheduler.get_last_lr()[0]:e}"""
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
                    if use_discriminator:
                        self.writer.add_scalar(
                            "d_lr_step",
                            self.disc_scheduler.get_last_lr()[0],
                            "train",
                            global_iter,
                        )

                if it % (len(loader) // 2) == 0:
                    self.model.eval()
                    bsz = xs.size(0)
                    if bsz == 1:  # need to be handle properly
                        continue
                    max_shard_size = min(bsz, 16)
                    xs, xs_recon = self.model_woddp.get_recon_imgs(
                        xs[:max_shard_size], xs_recon[:max_shard_size]
                    )
                    grid = (
                        torch.cat(
                            [
                                xs[: max_shard_size // 2],
                                xs_recon[: max_shard_size // 2],
                                xs[max_shard_size // 2 :],
                                xs_recon[max_shard_size // 2 :],
                            ],
                            dim=0,
                        )
                        .detach()
                        .cpu()
                        .float()
                    )
                    grid = torchvision.utils.make_grid(grid, nrow=max_shard_size // 2)
                    self.writer.add_image(
                        "reconstruction_step", grid, "train", global_iter
                    )
                    self.model.train()
        summary = accm.get_summary()
        summary["input"] = xs

        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_freq == 0:
            self.reconstruct(summary["input"], epoch, mode)
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
    def reconstruct(self, xs, epoch, mode="valid"):
        # do not write image when bs is 1
        if xs.size(0) == 1:
            return
        bsz = xs.size(0)
        max_shard_size = min((bsz // 2) * 2, 16)
        model = self.model_ema if "ema" in mode else self.model
        model.eval()
        xs_real = xs[:max_shard_size]
        xs_data = LabeledImageData(img=xs_real, condition=None, img_path=None)
        stage1_output: Stage1ModelOutput = self.model_woddp(xs_data)
        xs_recon = stage1_output.xs_recon
        xs_real, xs_recon = self.model_woddp.get_recon_imgs(xs_real, xs_recon)
        grid = torch.cat(
            [
                xs_real[: max_shard_size // 2],
                xs_recon[: max_shard_size // 2],
                xs_real[max_shard_size // 2 :],
                xs_recon[max_shard_size // 2 :],
            ],
            dim=0,
        )
        grid = (
            torchvision.utils.make_grid(grid, nrow=max_shard_size // 2)
            .detach()
            .cpu()
            .float()
        )
        self.writer.add_image("reconstruction", grid, mode, epoch)

    def _load_ckpt(self, optimizer, scheduler, epoch: int = -1, load_from_master=True):
        return super()._load_ckpt(
            optimizer, scheduler, epoch, additional_attr_to_load=("discriminator",)
        )

    def _load_model_only(
        self,
        load_path,
        additional_attr_to_load=("discriminator",),
        load_from_master=True,
    ):
        return super()._load_model_only(
            load_path, additional_attr_to_load, load_from_master
        )

    def save_ckpt(self, optimizer, scheduler, epoch):
        return super().save_ckpt(
            optimizer, scheduler, epoch, additional_attr_to_save=("discriminator",)
        )

    def _save_model_only(self, epoch, additional_attr_to_save=("discriminator",)):
        return super()._save_model_only(epoch, additional_attr_to_save)
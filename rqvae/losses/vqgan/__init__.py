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

from rqvae.optimizer import create_scheduler
from rqvae.optimizer.optimizer import create_resnet_optimizer

from .discriminator import NLayerDiscriminator, weights_init, DinoDiscriminator
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS
from .diffaug import DiffAug
import torch
def create_vqgan_loss(loss_config):

    disc_loss_type = loss_config.disc_loss
    if disc_loss_type == "hinge":
        disc_loss = hinge_d_loss
    elif disc_loss_type == "vanilla":
        disc_loss = vanilla_d_loss
    else:
        raise ValueError(f"Unknown GAN loss '{disc_loss_type}'.")

    gen_loss_type = loss_config.gen_loss
    if gen_loss_type == "vanilla":
        gen_loss = vanilla_g_loss
    else:
        raise ValueError(f"Unknown GAN loss '{gen_loss_type}'.")

    perceptual_loss = LPIPS()
    perceptual_loss.eval()

    return disc_loss, gen_loss, perceptual_loss


def create_discriminator_with_optimizer_scheduler(
    disc_config, steps_per_epoch, max_epoch, device, dtype, distenv=None, is_eval:bool = False
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, DiffAug]:
    use_vit = disc_config.arch.get("use_dino", False)
    if use_vit:
        model = DinoDiscriminator(
            device = 'cpu',
            dino_ckpt_path=disc_config.arch.dino_ckpt_path,
            ks = disc_config.arch.ks, # kernel size
            norm_type = disc_config.arch.norm_type, # normalization type
            norm_eps = 1e-6,
            using_spec_norm = True,
            key_depths = (2, 5, 8, 11) # fixed 
        ).to(device).to(dtype)
    else:
        model = NLayerDiscriminator(
            input_nc=disc_config.arch.in_channels,
            n_layers=disc_config.arch.num_layers,
            use_actnorm=disc_config.arch.use_actnorm,
            ndf=disc_config.arch.ndf,
        ).apply(weights_init).to(device).to(dtype)
    if not is_eval:
        optimizer = create_resnet_optimizer(model, disc_config.optimizer)
        scheduler = create_scheduler(
            optimizer,
            config=disc_config.optimizer,
            steps_per_epoch=steps_per_epoch,
            max_epoch=max_epoch,
            distenv=distenv,
        )
    else:
        optimizer = None
        scheduler = None
    augs = DiffAug()
    model.eval() # set model to eval mode
    return model, optimizer, scheduler, augs

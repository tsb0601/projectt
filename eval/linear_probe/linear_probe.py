# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import ViTMAEPreTrainedModel
import timm

from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import instantiate_from_config
from util.lars import LARS
from util.crop import RandomResizedCrop
from omegaconf import OmegaConf
from engine_finetune import train_one_epoch, evaluate
import torch_xla.core.xla_model as xm
from torch import nn
import torch_xla as xla
import torch_xla.distributed.xla_backend  # must be imported as init
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.runtime as xr

XLA_CACHE_PATH = os.environ.get("XLACACHE_PATH", "~/xla_compile/tmp")
os.makedirs(XLA_CACHE_PATH, exist_ok=True)
xr.initialize_cache(XLA_CACHE_PATH, readonly=False)


# add ../.. to the sys path
class head_model(nn.Module):
    def __init__(
        self, model_to_wrap: nn.Module, head: nn.Module, cls_token: bool = True
    ):
        super(head_model, self).__init__()
        self.model_to_wrap = model_to_wrap
        self.head = head
        self.cls_token = cls_token

    def forward(self, x):
        with torch.no_grad():
            x = self.model_to_wrap(x)
        if self.cls_token: # in this case x should be of shape [batch_size, seq_len, hidden_size]
            x = x[:, 0] # assure the dtype is the same as the input
        else: # do global pooling , x can be arbitrary shape but the first dim should be batch_size and the last dim should be hidden_size
            x = x.view(x.shape[0], -1, x.shape[-1]) # [batch_size, seq_len, hidden_size]
            # use global pooling on all tokens
            x = x.mean(dim=1) # [batch_size, hidden_size]
        #x = x.to(torch.float32) # use float32 for the head
        x = self.head(x)
        return x


sys.path.append(
    "../.."
)  # a hack to make sure one can import all the modules in the project


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE linear probing for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="weight decay (default: 0 for linear probe following MoCo v1)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=0.1,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch for training",
    )
    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden size")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument(
        "--model_config", default="", type=str, help="model configuration"
    )
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument(
        "--save_freq",
        default=10,
        type=int,
        help="save frequency of the model in epochs",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="xla:0", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")

    return parser


def main(args):
    misc.init_distributed_mode(args)
    device = xm.xla_device()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32    
    #torch.set_default_dtype(dtype) # set default dtype 
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    image_size = args.image_size
    # linear probe: weak augmentation
    #import numpy as np
    def custom_pil_to_tensor(pic):
        #first to numpy
        img = np.array(pic)
        #then to tensor
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).contiguous() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img   
    transform_train = transforms.Compose(
        [
            RandomResizedCrop(image_size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            custom_pil_to_tensor,
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224), interpolation=3),
            transforms.CenterCrop(image_size),
            #transforms.ToTensor(),
            custom_pil_to_tensor,
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=transform_train
    )
    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=transform_val
    )
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model_config = OmegaConf.load(args.model_config)
    # to dict
    model_dict = OmegaConf.to_container(model_config)
    model = instantiate_from_config(model_dict).to(device).to(dtype)# fix bfloat16
    # add head to the model
    linear_probe_head = torch.nn.Linear(args.hidden_size, args.nb_classes)
    trunc_normal_(linear_probe_head.weight, std=0.01)
    bn = torch.nn.BatchNorm1d(
        args.hidden_size, affine=False, eps=1e-6
    )  # use this could boost the performance
    head = torch.nn.Sequential(bn, linear_probe_head).to(device).to(dtype) # use float32 for the head
    model = head_model(model, head, not args.global_pool)
    if args.finetune and not args.eval:
        # model = head_model(model, head, args.cls_token)
        # if args.global_pool:
        #    raise NotImplementedError("Global pool not supported now")
        #    model.fc_norm = nn.LayerNorm(args.hidden_size, eps=1e-6, elementwise_affine=False) # a frozen layernorm
        #    model.norm = nn.Identity() # special judge for MAE model
        #    def forward_hook(module, input, output):
        #        #use global pooling of non-cls tokens
        #        latent = output[0] # the output is a tuple, latent: [batch_size, seq_len, hidden_size]
        #        x = latent[:, 1:, :].mean(dim=1) # [batch_size, hidden_size]
        #        x = model.fc_norm(x)
        #        x = model.head(x)
        #        return x
        # else:
        #    def forward_hook(module, input, output):
        #        x = output[0][:, 0]
        #        x = model.head(x) # [batch_size, nb_classes]
        #        print(x.shape)
        #        return x
        # model.register_forward_hook(forward_hook)
        if os.path.isfile(args.finetune):
            raise NotImplementedError(
                "Finetuning from a checkpoint is not supported now"
            )
            checkpoint = torch.load(args.finetune, map_location="cpu")
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint["model"]
            state_dict = model.state_dict()
            if not args.eval:
                for k in ["head.weight", "head.bias"]:
                    if (
                        k in checkpoint_model
                        and checkpoint_model[k].shape != state_dict[k].shape
                    ):
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

        # if args.global_pool:
        #    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        # trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, #eps=1e-6), model.head)
    #model = model.to(device).to(dtype)
    model.device = device
    model.dtype = dtype
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = DDP(model, gradient_as_bucket_view=True)
        model_without_ddp = model.module
    optimizer = torch.optim.AdamW(
        model_without_ddp.head.parameters(),
        betas = (0.9, 0.95),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    #optimizer = LARS(
    #    model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    #)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            dtype,
            epoch,
            loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and epoch % args.save_freq == 0:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")
        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

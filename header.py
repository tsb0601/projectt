import torch_xla.runtime as xr
import os
import argparse
import math
import torch
import torch.distributed as dist
import torch_xla as xla
import torch_xla.core.xla_model as xm
import wandb
from torch import nn
from typing import *
wandb_dir = os.environ.get("WANDB_DIR", None)
wandb_id = os.environ.get("WANDB_ID", None)
PROJECT_NAME = os.environ.get("WANDB_PROJECT", 'VAE-enhanced')
## For saving and loading checkpoints ##
CKPT_FOLDER = 'ep_{}-checkpoint/' # epoch
OPT_NAME = '{}-optimizer.pt' # rank
SCH_NAME = '{}-scheduler.pt' # rank
RNG_NAME = '{}-rng.pt' # rank
MODEL_NAME = '{}-model.pt' # rank
EMA_MODEL_NAME = '{}-ema-model.pt' # rank
ADDIONTIONAL_NAME = '{}-additional.pt' # rank
ACT_FOLDER = 'activation/' # epoch
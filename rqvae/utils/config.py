import os

from omegaconf import OmegaConf, DictConfig
from easydict import EasyDict as edict
import yaml



def easydict_to_dict(obj):
    if not isinstance(obj, edict):
        return obj
    else:
        return {k: easydict_to_dict(v) for k, v in obj.items()}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
    return config


def is_stage1_arch(arch_type):
    return not ('transformer' in arch_type)


def augment_arch_defaults(arch_config):

    if arch_config.type == 'dummy':
        arch_defaults = OmegaConf.create(
            {
                'ema': None,
            }
        )
    else:
        raise NotImplementedError

    return OmegaConf.merge(arch_defaults, arch_config)


def augment_optimizer_defaults(optim_config):

    defaults = OmegaConf.create(
        {
            'type': 'adamW',
            'max_gn': None,
            'warmup': {
                'mode': 'linear',
                'start_from_zero': (True if optim_config.warmup.epoch > 0 else False),
            },
        }
    )
    return OmegaConf.merge(defaults, optim_config)



def augment_dist_defaults(config, distenv):
    config = config.copy()

    local_batch_size = config.experiment.batch_size
    world_batch_size = distenv.world_size * local_batch_size
    total_batch_size = config.experiment.get('total_batch_size', world_batch_size)

    if total_batch_size % world_batch_size != 0:
        raise ValueError('total batch size must be divisible by world batch size')
    else:
        grad_accm_steps = total_batch_size // world_batch_size

    config.optimizer.grad_accm_steps = grad_accm_steps
    config.experiment.total_batch_size = total_batch_size

    return config


def config_setup(args, distenv, config_path, extra_args=()):

    if args.eval:
        config = load_config(config_path)
        if hasattr(args, 'test_batch_size'):
            config.experiment.batch_size = args.test_batch_size
        if not hasattr(config, 'seed'):
            config.seed = args.seed

    elif args.resume:
        config = load_config(config_path)
        if distenv.world_size != config.runtime.distenv.world_size:
            raise ValueError("world_size not identical to the resuming config")
        config.runtime = {'args': vars(args), 'distenv': distenv}

    else:  # training
        config_path = args.model_config
        config = load_config(config_path)

        extra_config = OmegaConf.from_dotlist(extra_args)
        config = OmegaConf.merge(config, extra_config)
        config = augment_dist_defaults(config, distenv)

        config.seed = args.seed
        config.runtime = {'args': vars(args), 'extra_config': extra_config, 'distenv': distenv}

    return config

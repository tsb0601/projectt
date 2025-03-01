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
from header import *
import rqvae.utils.dist as dist_utils
from rqvae.models import create_model
from rqvae.trainers import create_trainer
from rqvae.img_datasets import create_dataset
from rqvae.optimizer import create_optimizer, create_scheduler
from rqvae.utils.utils import compute_model_size, get_num_conv_linear_layers
from rqvae.utils.setup import setup , wandb_dir
#wandb.require('core')
import time
import torch_xla.distributed.xla_multiprocessing as xmp
xla._XLAC._xla_set_mat_mul_precision('highest') # set precision to high to assure accuracy
CACHE_DIR = '/home/tsb/.cache/xla_compile'
project_name = 'tmp'
cache_path = os.path.join(CACHE_DIR, project_name)
cache_path = os.environ.get('XLACACHE_PATH', cache_path)
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model-config', type=str, default='./configs/c10-igpt.yaml')
parser.add_argument('-r', '--result-path', type=str, default='./ckpt/tmp')
parser.add_argument('-l', '--load-path', type=str, default='')
parser.add_argument('-e', '--test-epoch', type=int, default=-1)
parser.add_argument('-p', '--postfix', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--reload-batch-size', type=int, default=None) # this will force to reload the dataset with the given batch size, should be used in inference
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='xla', choices=['xla'],type=str, help='distributed backend')
parser.add_argument('--timeout', type=int, default=120, help='time limit (s) to wait for other nodes in DDP')
parser.add_argument('--action', choices=['train', 'eval', 'cache_latent', 'gen', 'stat', 'dis'], default='train')
parser.add_argument('--exp', type=str, default=None) # experiment name
parser.add_argument('--resume', action='store_true')
parser.add_argument('--use_ddp', action='store_true')
parser.add_argument('--use_autocast', action='store_true')
parser.add_argument('--do_online_eval', action='store_true') # if we want to do online eval for FID
parser.add_argument('--fid_gt_act_path', type=str, default='ckpt_gcs/acts/val_256_act.npz') # GT activations for FID
def main(rank, args, extra_args):
    start = time.time()
    global cache_path
    args.rank = rank
    config, logger, writer = setup(args, extra_args)
    xm.master_print(f'[!]XLACACHE_PATH: {cache_path}')
    os.makedirs(cache_path, exist_ok=True)
    if not xla._XLAC._xla_computation_cache_is_initialized(): # only initialize once
        # TODO: add a lock to prevent multiple processes from initializing the cache
        xr.initialize_cache(cache_path, readonly=False)
    distenv = config.runtime.distenv
    if distenv.master and wandb_dir:
        wandb.save(str(args.model_config)) # save the config file
    device = xm.xla_device()
    print(f'Using device: {device}')
    xm.master_print(f'loading dataset of {config.dataset.type}...')
    is_eval = args.action != 'train' # if not training we won't need to do optimizer and stuff
    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)
    xm.master_print(f'loaded dataset of {config.dataset.type}...')
    xm.master_print(f'train dataset size: {len(dataset_trn)}, valid dataset size: {len(dataset_val)}')
    xm.master_print(f'world_size: {distenv.world_size}, local_rank: {distenv.local_rank}, node_rank: {distenv.world_rank}')
    model, model_ema = create_model(config.arch, ema=config.arch.ema, is_master=distenv.master)
    model.to(device)
    if model_ema:
        model_ema.to(device)
    xm.master_print(f'[!]model created, use_ema: {model_ema is not None}, ema_decay: {config.arch.ema if model_ema is not None else None}, use_ddp: {args.use_ddp}')
    trainer = create_trainer(config)
    xm.master_print(f'[!]trainer created')
    if args.reload_batch_size:
        config.experiment.batch_size = args.reload_batch_size
        config.experiment.actual_batch_size = config.experiment.batch_size * distenv.world_size * config.experiment.accu_step
    elif config.experiment.get('actual_batch_size', None) is not None:
        config.experiment.batch_size = config.experiment.actual_batch_size // (distenv.world_size * config.experiment.accu_step)
        assert config.experiment.batch_size * distenv.world_size * config.experiment.accu_step == config.experiment.actual_batch_size, f'actual_batch_size: {config.experiment.actual_batch_size} cannot be divided by world_size: {distenv.world_size} and accu_step: {config.experiment.accu_step}'
    else:
        actual_batch_size = config.experiment.batch_size * distenv.world_size * config.experiment.accu_step
        config.experiment.actual_batch_size = actual_batch_size
    actual_batch_size = config.experiment.actual_batch_size
    train_epochs = config.experiment.epochs
    steps_per_epoch = math.ceil(len(dataset_trn) / actual_batch_size)
    epoch_st = 0
    xm.master_print(f'[!] micro_batch_size_per_core: {config.experiment.batch_size}, accu_step: {config.experiment.accu_step}, actual_batch_size: {actual_batch_size}, steps_per_epoch: {steps_per_epoch}')
    if distenv.master:
        logger.info(f'#conv+linear layers: {get_num_conv_linear_layers(model)}')
    use_optim = not is_eval
    xm.master_print(f'[!]use_optim: {use_optim}')
    if use_optim:
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(
            optimizer, config.optimizer, steps_per_epoch,
            config.experiment.epochs, distenv
        )
    disc_state_dict = None
    xm.master_print(f'[!]model loaded')
    if distenv.master:
        print(model)
        print(f'[!]model dtype: {next(model.parameters()).dtype}')
        compute_model_size(model, logger)
    if distenv.master and use_optim:
        logger.info(optimizer.__repr__())
    model = dist_utils.dataparallel_and_sync(distenv, model)
    if model_ema:
        model_ema = dist_utils.dataparallel_and_sync(distenv, model_ema)
    trainer = trainer(model, model_ema, dataset_trn, dataset_val, config, writer,
                      device, distenv, disc_state_dict=disc_state_dict, eval = is_eval ,use_ddp=args.use_ddp, use_autocast=args.use_autocast,do_online_eval=args.do_online_eval, fid_gt_act_path=args.fid_gt_act_path) 
    xm.master_print(f'[!]trainer created')
    if not args.load_path == '' and os.path.exists(args.load_path):
        if args.resume and not is_eval:
            trainer._load_ckpt(args.load_path, optimizer, scheduler)
            #load_path should end with /ep_{epoch}-checkpoint/, we parse the epoch from the path
            epoch_st = os.path.basename(args.load_path).split('-')[0].split('_')[-1]
            if epoch_st == 'last':
                xm.master_print(f'[!]model already trained complete, exit')
                exit()
            epoch_st = int(epoch_st) # actual epoch to start
        else:
            trainer._load_model_only(args.load_path,additional_attr_to_load= ())
        xm.master_print(f'[!]model loaded from {args.load_path} with resume: {args.resume}')
        xm.mark_step()
    xm.master_print(f'[!]all trainer config created, start for {train_epochs - epoch_st} epochs from ep {epoch_st} to ep {train_epochs}')
    if args.action == 'cache_latent':
        train_save_path = os.path.join(args.result_path, 'train')
        valid_save_path = os.path.join(args.result_path, 'valid')
        os.makedirs(train_save_path, exist_ok=True)
        os.makedirs(valid_save_path, exist_ok=True)
        xm.master_print(f'[!]caching latent for train and valid dataset')
        trainer.cache_latent(feature_path=train_save_path, valid=False)
        xm.master_print(f'[!]caching latent for valid dataset')
        trainer.cache_latent(feature_path=valid_save_path, valid =True)
    elif args.action == 'gen':
        trainer.batch_infer(valid=True, save_root=args.result_path)
    elif args.action == 'eval':
        trainer.eval(valid=True, verbose=True)
    elif args.action == 'stat':
        bn = trainer.calculate_mean_and_std(valid = False) # calculate mean and std for the training dataset
        running_mean, running_var = bn.running_mean, bn.running_var
        xm.master_print(f'[!]running_mean: {running_mean.shape}, running_var: {running_var.shape}')
        # only save the connector
        if distenv.master:
            connecter = trainer.model_woddp.connector.cpu()
            connector_path = os.path.join(args.result_path, 'connector.pt')
            # save the whole bn instead of state_dict
            torch.save(connecter, connector_path)
            xm.master_print(f'[!]connector saved in {connector_path}')
    elif args.action == 'dis':
        distance = trainer.calculate_distance(valid = True) # calculate distance for the validation dataset
        xm.master_print(f'[!]distance: {distance}')
    else:
        trainer.run_epoch(optimizer, scheduler, epoch_st)
    xm.master_print(f'[!]finished in {time.time() - start} seconds')
    if distenv.master:
        writer.close()  # may prevent from a file stable error in brain cloud..
        #if wandb_dir:
        #    wandb.finish()
    xm.master_print(f'[!]finished in {time.time() - start} seconds')
    xm.master_print(f'[!]Results saved in {config.result_path}')
    if args.use_ddp:
        dist.destroy_process_group()
    xm.rendezvous('done')
    
if __name__ == '__main__':
    args, extra_args = parser.parse_known_args()
    args.eval = args.action != 'train' # sp judge for backward compatibility
    xmp.spawn(main, args=(args, extra_args), start_method='fork')
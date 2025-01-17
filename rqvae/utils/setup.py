from datetime import datetime
import logging
import inspect
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from .writer import Writer
from .config import config_setup
from .dist import initialize as dist_init
from header import wandb_dir, PROJECT_NAME, wandb_id, xm, wandb # import wandb related variables
import torch_xla.core.xla_model as xm
#get home dir
def logger_setup(log_path, eval=False):
    global wandb_dir, PROJECT_NAME
    if wandb_dir:
        original_log_path = log_path
        log_path = os.path.join(Path.home(),'tmp', os.path.basename(log_path))
        os.makedirs(log_path, exist_ok=True)
    log_fname = os.path.join(log_path, 'val.log' if eval else 'train.log')

    for hdlr in logging.root.handlers:
        logging.root.removeHandler(hdlr)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname), logging.StreamHandler()
        ],
    )
    main_filename, *_ = inspect.getframeinfo(inspect.currentframe().f_back.f_back)

    logger = logging.getLogger(Path(main_filename).name)
    if wandb_dir:
        # find the parent directory of log_path
        resume = "allow" if wandb_id else None
        #wandb.tensorboard.patch(root_logdir=os.path.join(log_path,'train'), pytorch=True, tensorboard_x=False)
        wandb.init(project=PROJECT_NAME, sync_tensorboard=True, dir=log_path, name=os.path.basename(original_log_path), id=wandb_id, resume=resume,reinit=True) # set reinit= True so we don't have to call finish
        xm.master_print(f'wandb initialized with project: {PROJECT_NAME}, log_path: {log_path}, {"resume from id: " + wandb_id if wandb_id else ""}')
    writer = Writer(log_path)

    return logger, writer

def setup_quick(args):
    """
    meaning of args.result_path:
        - if args.eval, directory where the model is
        - if args.resume, no meaning
        - otherwise, path to store the logs

    Returns:
        config, logger, writer
    """

    distenv = dist_init(args)





def setup(args, extra_args=()):
    """
    meaning of args.result_path:
        - if args.eval, directory where the model is
        - if args.resume, no meaning
        - otherwise, path to store the logs

    Returns:
        config, logger, writer
    """

    distenv = dist_init(args)

    args.result_path = Path(args.result_path).absolute().as_posix()
    args.model_config = Path(args.model_config).absolute().resolve().as_posix()

    now = datetime.now().strftime('%d%m%Y_%H%M%S') if args.exp is None else args.exp + '_' + datetime.now().strftime('%d%m%Y_%H%M%S')
    config_path = Path(args.model_config).absolute()
    if args.eval:
        log_path = Path('./logs/tmp').joinpath(now)
    elif args.resume:
        task_name = config_path.stem
        config_path = Path(args.load_path).parent.joinpath('config.yaml')
        log_path = Path(args.result_path).joinpath(task_name, now)
    else:
        task_name = config_path.stem
        if args.postfix:
            task_name += f'__{args.postfix}'
        log_path = Path(args.result_path).joinpath(task_name, now)

    config = config_setup(args, distenv, config_path, extra_args=extra_args)
    config.result_path = log_path.absolute().resolve().as_posix()
    config.load_path = args.load_path
    if distenv.master:
        if not log_path.exists():
            os.makedirs(log_path)
        logger, writer = logger_setup(log_path)
        logger.info(distenv)
        logger.info(f'log_path: {log_path}')
        logger.info('\n' + OmegaConf.to_yaml(config))
        OmegaConf.save(config, log_path.joinpath('config.yaml'))
        src_dir = Path(os.getcwd()).joinpath('rqvae')
        shutil.copytree(src_dir, log_path.joinpath('rqvae'))
        logger.info(f'source copied to {log_path}/rqvae')
    else:
        logger, writer, log_path = None, None, None

    return config, logger, writer

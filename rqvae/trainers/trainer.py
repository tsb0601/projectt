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

import os
import logging
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch_xla.distributed.parallel_loader import ParallelLoader
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from PIL import Image
from zmq import device
from header import *
from rqvae.img_datasets.interfaces import LabeledImageData
from rqvae.models.interfaces import Stage1Encodings, Stage1ModelOutput, Stage1ModelWrapper, Stage2ModelOutput, Stage2ModelWrapper, XLA_Model
from torch_xla.amp import autocast
from contextlib import nullcontext
from rqvae.metrics.fid import InceptionWrapper, frechet_distance, Inception_Score, InceptionV3
from rqvae.utils.monitor import norm_tracker
logger = logging.getLogger(__name__)
DEBUG = bool(os.environ.get("DEBUG", 0))

class OnlineVarianceTracker:
    def __init__(self):
        self.n = 0          # total number of samples seen so far
        self.mean = None    # running mean (D,)
        self.M2 = None      # running sum of squared differences (D,)
    @torch.no_grad()
    def update(self, X: torch.Tensor) -> None:
        """
        Update the running statistics with a mini-batch of data X (N, D).
        We'll treat the incoming batch as a separate "dataset" and merge it
        with our running statistics in one step.
        """
        N = X.shape[0]
        if N == 0:
            return  # No new data in this batch

        # Compute batch mean and M2
        batch_mean = X.mean(dim=0)
        # Compute sum of squared deviations from batch mean
        # (X - batch_mean) is (N, D) -> squared -> sum over N -> (D,)
        batch_M2 = ((X - batch_mean)**2).sum(dim=0)

        if self.mean is None:
            # If no previous data, just take batch stats
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = N
        else:
            # Merge with existing stats
            n_old = self.n
            n_new = n_old + N

            delta = batch_mean - self.mean
            # Update mean
            new_mean = self.mean + delta * (N / n_new)

            # Update M2
            # M2_new = M2_old + M2_batch + delta^2 * (n_old * N / n_new)
            M2_new = self.M2 + batch_M2 + (delta**2) * (n_old * N / n_new)

            # Assign updated stats
            self.mean = new_mean
            self.M2 = M2_new
            self.n = n_new

    def trace_sigma(self) -> torch.Tensor:
        """
        Compute and return the trace of the covariance matrix.

        Σ = M2 / n, so Tr(Σ) = sum of variances = sum(M2 / n).
        """
        if self.n == 0:
            return torch.tensor(0.0, dtype=torch.float)
        var = self.M2 / self.n
        return var
    def __call__(self, X: torch.Tensor) -> None:
        self.update(X)
from scipy.special import gammaln
class OnlineEnergyDistance:
    def __init__(self, d, sample_size=2):
        """
        Initialize the online tracker for energy distance.
        
        Parameters:
        d : int
            Dimensionality of the data.
        sample_size : int
            Number of random samples used for approximating ||X - X'|| in each mini-batch.
        """
        self.n = 0  # Total number of samples processed
        self.total_pairwise_dist = 0.0  # Sum of pairwise distances ||X - X'||
        self.total_cross_dist = 0.0  # Sum of cross distances ||X - Y||
        self.d = d  # Dimensionality of the data
        self.sample_size = sample_size  # Number of samples for pairwise distance approximation

        # Precompute E[||Y - Y'||] for standard Gaussian
        #self.constant_dist = 2 * gamma((d + 1) / 2) / gamma(d / 2)
        gammaln_ = torch.tensor(gammaln((d+1) / 2) - gammaln(d / 2), dtype=torch.float64)
        self.constant_dist = 2 * torch.exp(gammaln_)
        print(f"Constant distance: {self.constant_dist}")
    
    def update(self, X):
        """
        Update the energy distance with a new mini-batch of samples X.
        
        Parameters:
        X : torch.Tensor of shape (N, D)
            A batch of samples from the distribution P.
        """
        N, d = X.shape
        if d != self.d:
            raise ValueError("Dimensionality mismatch")

        # Pairwise distances within a sampled subset of the batch
        rand_index = torch.randperm(N)
        sample_indices = rand_index[:self.sample_size]
        second_indices = rand_index[-self.sample_size:]
        X_sample = X[sample_indices]
        X_sample_prime = X[second_indices]
        pairwise_distances = torch.cdist(X_sample, X_sample_prime, p=2)
        #print(f"Pairwise distances: {pairwise_distances}")
        self.total_pairwise_dist += pairwise_distances.mean().item()

        # Cross distances between batch and standard Gaussian
        Y = torch.randn_like(X_sample)  # Samples from N(0, I)
        cross_distances = torch.cdist(X_sample, Y, p=2)
        #print(f"Cross distances: {cross_distances}")
        self.total_cross_dist += cross_distances.mean().item()
        #print(f"Cross distances: {cross_distances.mean().item()}, Pairwise distances: {pairwise_distances.mean().item()}")
        # Update total sample count
        self.n += 1

    def energy_distance(self):
        """
        Compute the energy distance to the standard Gaussian.
        """
        if self.n == 0:
            return 0.0
        # Compute the final energy distance
        ED = 2 * (self.total_cross_dist / self.n) - (self.total_pairwise_dist / self.n) - self.constant_dist
        return ED
    def __call__(self, X):
        self.update(X)
class TrainerTemplate:
    def __init__(
        self,
        model,
        model_ema,
        dataset_trn,
        dataset_val,
        config,
        writer,
        device,
        distenv,
        model_aux=None,
        *,
        disc_state_dict=None,  # only used in VQGAN trainer
        eval:bool = False,
        use_ddp:bool = False,
        use_autocast:bool = False,
        do_online_eval:bool = False,
        fid_gt_act_path:str = None,
    ):
        super().__init__()
        global wandb_dir
        num_workers = 4
        self.model = model
        self.use_wandb = wandb.run is not None
        self.model_ema = model_ema
        self.model_woddp = model.module if use_ddp else model
        self.model_ema_woddp = model_ema.module if use_ddp and (model_ema is not None) else model_ema
        self.model_aux = model_aux
        self.use_ddp = use_ddp
        self.dtype = self.model_woddp.get_last_layer().dtype
        self.config = config
        self.writer = writer
        self.device = device
        self.is_eval = eval
        self.use_autocast = use_autocast
        self.distenv = distenv
        self.accu_step = config.experiment.accu_step
        self.actual_batch_size = config.experiment.actual_batch_size
        self.dataset_trn = dataset_trn
        self.dataset_val = dataset_val
        self.do_online_eval = do_online_eval
        self.fid_gt_act_path = fid_gt_act_path
        if config.get("optimizer", None) is not None:
            self.clip_grad_norm = config.optimizer.get("clip_grad_norm", 0)
        else:
            self.clip_grad_norm = 0
        self.norm_tracker = norm_tracker(self.model.parameters(), max_norm=self.clip_grad_norm)
        if do_online_eval:
            assert fid_gt_act_path is not None, "fid_gt_act_path should be provided for do_online_eval"
            self.inception_model = InceptionWrapper([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device)
            self.fid_gt_act = np.load(fid_gt_act_path)['act'] if self.distenv.master else None
        self.sampler_trn = torch.utils.data.distributed.DistributedSampler(
            self.dataset_trn,
            num_replicas=self.distenv.world_size,
            rank=self.distenv.world_rank,
            shuffle=True,
            seed=self.config.seed,
        )
        self.loader_trn = DataLoader(
            self.dataset_trn,
            sampler=self.sampler_trn,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
            drop_last=True, # very important for xla to avoid dynamic shape (sometimes)
            collate_fn=self.dataset_trn.collate_fn if hasattr(self.dataset_trn, 'collate_fn') else None
        )

        self.sampler_val = torch.utils.data.distributed.DistributedSampler(
            self.dataset_val,
            num_replicas=self.distenv.world_size,
            rank=self.distenv.world_rank,
            shuffle=False,
        )
        self.loader_val = DataLoader(
            self.dataset_val,
            sampler=self.sampler_val,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
            drop_last=False, # drop_last=True to avoid dynamic shape
            collate_fn=self.dataset_val.collate_fn if hasattr(self.dataset_val, 'collate_fn') else None
        )
        if self.distenv.master:
            logger.info(
                f"train dataset size: {len(dataset_trn)}, valid dataset size: {len(dataset_val)}"
            )
        if xm.xrt_world_size() > 1 and self.distenv.TPU:
            if use_ddp:
                assert (
                    dist.is_initialized()
                ), "Distributed training is not initialized when using multiple xla device"
            self.parallel_loader_trn = ParallelLoader(self.loader_trn, [self.device])
            self.parallel_loader_val = ParallelLoader(
                self.loader_val, [self.device]
            )  # in xla we use pl
        else:
            raise NotImplementedError
    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        raise NotImplementedError

    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        raise NotImplementedError
    
    def wrap_loader(self, split:str = 'train'):
        assert split in ['train', 'valid'], f"split should be either 'train' or 'valid', but got {split}"
        loader = self.loader_trn if split == 'train' else self.loader_val
        if self.distenv.TPU:
            loader = ParallelLoader(loader, [self.device]).per_device_loader(self.device)
        return loader
    def tensor_image_to_numpy(self, tensor_im: torch.Tensor) -> np.ndarray:
        """
        tensor should be in range [0, 1]
        """
        len_shape = len(tensor_im.shape)
        if len_shape == 3:
            tensor_im = tensor_im.unsqueeze(0)
        tensor_im = tensor_im * 2 - 1
        tensor_im = torch.clamp(tensor_im * 127.5 + 128, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        if len_shape == 3:
            tensor_im = tensor_im.squeeze(0)
        return tensor_im
    @torch.no_grad()
    def calculate_distance(self, ema: bool = False, valid:bool = True):
        """
        calculate the generated latent distribution's Wasserstein-2 distance to a Gaussian distribution
        """
        self.model.eval()
        if self.model_ema is not None:
            self.model_ema.eval()
        model = self.model_woddp if not ema or self.model_ema is None else self.model_ema_woddp
        loader = self.wrap_loader('valid' if valid else 'train')
        pbar = tqdm(enumerate(loader), desc='Calculating distance', disable=not self.distenv.master,total=len(loader))
        
        for it, inputs in pbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            model: Union[Stage1ModelWrapper, Stage2ModelWrapper] = model
            with autocast(device=self.device) if self.use_autocast else nullcontext():
                if isinstance(model, Stage1ModelWrapper):
                    outputs:Stage1Encodings = model.encode(inputs)
                    zs = outputs.zs
                else:
                    outputs:Stage2ModelOutput = model.infer(inputs) # infer the latent distribution
                    zs = outputs.zs_pred
            zs = zs.reshape(outputs.zs.shape[0], -1) # (B, D)
            if it == 0:
                tracker = OnlineEnergyDistance(zs.shape[-1], sample_size=int(.3 * zs.shape[0]))
            #print(zs.shape)
            tracker(zs)
        sigma = tracker.energy_distance()
        sigma = torch.tensor(sigma, dtype=torch.float64, device=self.device).view(1) # (1,)
        # all gather
        sigma = xm.all_gather(sigma, dim=0, pin_layout=True)
        sigma = sigma.cpu().numpy()
        if self.distenv.master:
            sigma = sigma.mean()
            return sigma
            #estimated_sigma = np.mean(sigma, axis=0) # (D,)
            #print(f"Estimated sigma: {estimated_sigma.sum()}, sqrt(sigma): {np.sqrt(estimated_sigma).sum()}")
            ## calculate W2 distance to Gaussian: W2 = Tr(Σ) - 2Tr(Σ**0.5) + d
            #d = zs.shape[-1] # dimension
            #W2 = np.sum(estimated_sigma) - 2 * np.sum(np.sqrt(estimated_sigma)) + d
            #return W2
        return 0.0
    @torch.no_grad()
    def batch_infer(self, ema: bool = False, valid:bool = True , save_root:str=None, test_fid:bool = False, epoch:int = 0):
        #assert os.path.exists(save_root), f"save_root {save_root} does not exist"
        if save_root is not None and self.distenv.master:
            assert os.path.exists(save_root), f"save_root {save_root} does not exist"
        self.model.eval() 
        if self.model_ema is not None:
            self.model_ema.eval()
        model = self.model_woddp if not ema or self.model_ema is None else self.model_ema_woddp
        loader = self.wrap_loader('valid' if valid else 'train')
        pbar = tqdm(enumerate(loader), desc='Inferencing' if not test_fid else 'Testing FID', disable=not self.distenv.master,total=len(loader))
        if test_fid:
            #if self.distenv.master:
            fid_gt_act = self.fid_gt_act if self.distenv.master else None
            inception_model = self.inception_model
            dataset = self.dataset_val if valid else self.dataset_trn
            inception_acts = []
            inception_logits = []
            inception_model.eval()
            st_idx = 0
        for it, inputs in pbar:
            inputs: LabeledImageData
            inputs._to(self.device)._to(self.dtype)
            img_paths = inputs.img_path
            with autocast(device=self.device) if self.use_autocast else nullcontext():
                outputs:Stage1ModelOutput = model.infer(inputs) 
            xs_recon_or_gen = outputs.xs_recon.detach().clone().float() # destroy the graph
            xm.mark_step()
            if test_fid: # we want float32
                # convert to uint8 then back
                xs_recon_or_gen = self.tensor_image_to_numpy(xs_recon_or_gen)
                xs_recon_or_gen = torch.from_numpy(xs_recon_or_gen).to(torch.float32).to(self.device) / 255. #do the exact same thing as image gen pipeline
                xs_recon_or_gen = xs_recon_or_gen.permute(0, 3, 1, 2) # (B, C, H, W)
                incep_act, incep_logits = inception_model.get_logits(xs_recon_or_gen) # (B, 2048)
                inception_acts.append(incep_act)
                inception_logits.append(torch.nn.functional.softmax(incep_logits, dim=-1))
                st_idx += len(incep_act)
            if save_root is not None and not test_fid:
                for i, img_path in enumerate(img_paths):
                    img_name = os.path.basename(img_path)
                    # change the suffix to png
                    img_name = img_name.split('.')[0] + '.png'
                    save_path = os.path.join(save_root, img_name)
                    img = self.tensor_image_to_numpy(xs_recon_or_gen[i]) 
                    img = Image.fromarray(img)
                    img.save(save_path)  
            xm.mark_step()
        if test_fid:
            # all gather
            inception_acts = torch.cat(inception_acts, dim=0).to(self.device)
            inception_logits = torch.cat(inception_logits, dim=0).to(self.device)
            inception_acts = xm.all_gather(inception_acts, dim=0, pin_layout=True) # (N, 2048)
            inception_logits = xm.all_gather(inception_logits, dim=0, pin_layout=True) # (N, 1008)
            # if len(inception_acts) > len(self.dataset_val) we choose the first len(self.dataset_val) samples
            inception_acts = inception_acts[:len(dataset)]
            inception_logits = inception_logits[:len(dataset)] #
            # do fid on master
            inception_acts = inception_acts.cpu().numpy()
            inception_logits = inception_logits.cpu().float()
            if self.distenv.master:    
                mu_gt = np.mean(fid_gt_act, axis=0)
                sigma_gt = np.cov(fid_gt_act, rowvar=False)
                mu = np.mean(inception_acts, axis=0)
                sigma = np.cov(inception_acts, rowvar=False)
                fid = frechet_distance(mu_gt, sigma_gt, mu, sigma)
                IS_value, IS_std = Inception_Score(inception_logits)
                if save_root is not None:
                    np.savez(os.path.join(save_root, f'ep{epoch}_act.npz'), 
                            act = inception_acts,
                            logits = inception_logits.cpu().numpy()
                    )
                return (fid, IS_value, IS_std)
        return ()
    def logging(self, *args, **kwargs):
        raise NotImplementedError
    def run_epoch(self, optimizer=None, scheduler=None, epoch_st=0):
        global CKPT_FOLDER, ACT_FOLDER
        for i in range(epoch_st, self.config.experiment.epochs):
            self.sampler_trn.set_epoch(i)
            # next epoch is i+1
            if (i % self.config.experiment.save_ckpt_freq) == 0 and (i != self.config.experiment.epochs - 1) and (i != epoch_st):
                self.save_ckpt(optimizer, scheduler, i) 
                # next epoch is i+1
            if i % self.config.experiment.test_freq == 0 or i == self.config.experiment.epochs - 1: # do validation every test_freq or last epoch
                if self.do_online_eval:
                    act_save_path = os.path.join(self.config.result_path, ACT_FOLDER)
                    if self.distenv.master:
                        os.makedirs(act_save_path, exist_ok=True)
                    stats = self.batch_infer(ema=False, valid=True, save_root=act_save_path, test_fid=True, epoch = i)
                    if self.distenv.master:
                        fid, IS_value, IS_std = stats
                        print(f"Epoch {i} FID: {fid}, IS: {IS_value} +/- {IS_std}")
                        self.writer.add_scalar("metrics/FID", fid, "valid",i )
                        self.writer.add_scalar("metrics/IS", IS_value, "valid",i ) 
                        self.writer.add_scalar("metrics/IS_std", IS_std, "valid",i )
                    if self.model_ema is not None:
                        ema_stats = self.batch_infer(ema=True, valid=True, save_root=act_save_path, test_fid=True, epoch = i)
                        if self.distenv.master:
                            fid, IS_value, IS_std = ema_stats
                            print(f"Epoch {i} EMA FID: {fid}, IS: {IS_value} +/- {IS_std}")
                            self.writer.add_scalar("metrics/FID", fid, "valid_ema",i )
                            self.writer.add_scalar("metrics/IS", IS_value, "valid_ema",i ) 
                            self.writer.add_scalar("metrics/IS_std", IS_std, "valid_ema",i )
                summary_val = self.eval(epoch=i, valid=True, verbose=True)
                if self.model_ema is not None:
                    summary_val_ema = self.eval(ema=True, epoch=i, valid=True, verbose=True)
            summary_trn = self.train(optimizer, scheduler, None, epoch=i) # we do not use scaler in TPU
            if self.distenv.master:
                self.logging(
                    summary_trn, scheduler=scheduler, epoch=i, mode="train"
                )
                if i % self.config.experiment.test_freq == 0:
                    self.logging(
                        summary_val, scheduler=scheduler, epoch=i, mode="valid"
                    )
                    if self.model_ema is not None:
                        self.logging(
                            summary_val_ema,
                            scheduler=scheduler,
                            epoch=i,
                            mode="valid_ema",
                        )
            xm.rendezvous(
                "epoch_sync"
            )  # make sure we save the model properly without stuck
        self.save_ckpt(optimizer, scheduler, -1 )# last ckpt
    def _load_model_only(self, load_path, additional_attr_to_load:tuple = (), load_from_master:bool = True):
        global CKPT_FOLDER, MODEL_NAME, EMA_MODEL_NAME, ADDIONTIONAL_NAME
        rank = self.distenv.world_rank if not load_from_master else 0 # load from master (rank:0)
        model_path = os.path.join(load_path, MODEL_NAME.format(rank))
        model_weight = torch.load(model_path) # to xla directly
        self.model_woddp.load_state_dict(model_weight)
        if self.model_ema:
            ema_model_path = os.path.join(load_path, EMA_MODEL_NAME.format(rank))
            if os.path.exists(ema_model_path):
                ema_model_weight = torch.load(ema_model_path)
                keys = self.model_ema_woddp.load_state_dict(ema_model_weight, strict=False)
                xm.master_print(f"[!] EMA model loaded with keys: {keys}")
            else:
                xm.master_print(f"[!] EMA model path {ema_model_path} does not exist, skip loading EMA model")
        if len(additional_attr_to_load) == 0:
            return
        additional_path = os.path.join(load_path, ADDIONTIONAL_NAME.format(rank))
        additional_attr_ckpt = torch.load(additional_path)
        for attr in additional_attr_to_load:
            assert attr in additional_attr_ckpt, f"additional_attr_to_load {attr} not in additional_attr_ckpt"
            assert hasattr(self, attr), f"self does not have attribute {attr}"
            target_module = getattr(self, attr).module if self.use_ddp else getattr(self, attr)
            target_module.load_state_dict(additional_attr_ckpt[attr]) 
    def _save_model_only(self, epoch, additional_attr_to_save:tuple = ()):
        global CKPT_FOLDER, MODEL_NAME, EMA_MODEL_NAME, ADDIONTIONAL_NAME
        rank = self.distenv.world_rank
        # only save from master
        if not self.distenv.master:
            return
        epoch = 'last' if epoch == -1 else epoch
        ckpt_folder = os.path.join(self.config.result_path , CKPT_FOLDER.format(epoch))
        os.makedirs(ckpt_folder, exist_ok=True)
        model_path = os.path.join(ckpt_folder, MODEL_NAME.format(rank))
        additional_path = os.path.join(ckpt_folder, ADDIONTIONAL_NAME.format(rank))
        model_weight = self.sync_and_to_cpu(self.model_woddp.state_dict())
        torch.save(model_weight, model_path)
        additional_attr_ckpt = {}  
        for attr in additional_attr_to_save:
            target_module = getattr(self, attr).module if self.use_ddp else getattr(self, attr)
            additional_attr_ckpt[attr] = self.sync_and_to_cpu(target_module.state_dict())
        if len(additional_attr_ckpt) > 0:
            torch.save(additional_attr_ckpt, additional_path)
        if self.model_ema:
            ema_model_path = os.path.join(ckpt_folder, EMA_MODEL_NAME.format(rank))
            ema_model_weight = self.sync_and_to_cpu(self.model_ema_woddp.state_dict())
            torch.save(ema_model_weight, ema_model_path)
    def _load_ckpt(self,load_path, optimizer, scheduler, additional_attr_to_load:tuple = (), load_from_master:bool = True):
        global CKPT_FOLDER, MODEL_NAME, OPT_NAME, SCH_NAME, ADDIONTIONAL_NAME, EMA_MODEL_NAME, RNG_NAME
        rank = self.distenv.world_rank if not load_from_master else 0
        ckpt_folder = load_path
        model_path = os.path.join(ckpt_folder, MODEL_NAME.format(rank))
        opt_path = os.path.join(ckpt_folder, OPT_NAME.format(rank))
        sch_path = os.path.join(ckpt_folder, SCH_NAME.format(rank))
        rng_path = os.path.join(ckpt_folder, RNG_NAME.format(self.distenv.world_rank)) # we assume rng is saved for each rank
        if not os.path.exists(rng_path):
            rng_path = os.path.join(ckpt_folder, RNG_NAME.format(0)) # load from master
        model_weight = torch.load(model_path)
        optimizer_weight = torch.load(opt_path)
        scheduler_weight = torch.load(sch_path)
        self.model_woddp.load_state_dict(model_weight)
        optimizer.load_state_dict(optimizer_weight)
        scheduler.load_state_dict(scheduler_weight)
        if os.path.exists(rng_path):
            rng_state = torch.load(rng_path)
            torch.set_rng_state(rng_state['torch'])
            np.random.set_state(rng_state['numpy'])
            random.setstate(rng_state['random'])
            xm.set_rng_state(rng_state['xm'])
        if len(additional_attr_to_load) == 0:
            return
        additional_path = os.path.join(ckpt_folder, ADDIONTIONAL_NAME.format(rank))
        additional_attr_ckpt = torch.load(additional_path)
        for attr in additional_attr_to_load:
            assert attr in additional_attr_ckpt, f"additional_attr_to_load {attr} not in additional_attr_ckpt"
            assert hasattr(self, attr), f"self does not have attribute {attr}"
            target_module = getattr(self, attr).module if self.use_ddp else getattr(self, attr)
            target_module.load_state_dict(additional_attr_ckpt[attr])
        if self.model_ema:
            ema_model_path = os.path.join(ckpt_folder, EMA_MODEL_NAME.format(rank))
            if os.path.exists(ema_model_path):
                ema_model_weight = torch.load(ema_model_path)
                self.model_ema_woddp.load_state_dict(ema_model_weight)
            else:
                xm.master_print(f"[!] EMA model path {ema_model_path} does not exist, skip loading EMA model")
    def sync_and_to_cpu(self, state_dict, key_match:Optional[dict] = None):
        """
        """
        def convert_fn(item):
            if isinstance(item, torch.Tensor):
                item = xm._maybe_convert_to_cpu(item).to(torch.float32) # to cpu and float32
                return item
            elif isinstance(item, dict):
                return {k: convert_fn(v) for k,v in item.items()}
            elif isinstance(item, list):
                return [convert_fn(v) for v in item]
            elif isinstance(item, tuple):
                return tuple(convert_fn(v) for v in item)
            else:
                return item
        state_dict = {
            k: convert_fn(v) for k,v in state_dict.items() if key_match is None or k in key_match
        }
        return state_dict
    def save_ckpt(self, optimizer, scheduler, epoch, additional_attr_to_save:tuple = (), master_only:bool = True):
        global CKPT_FOLDER, MODEL_NAME, OPT_NAME, SCH_NAME, ADDIONTIONAL_NAME, EMA_MODEL_NAME, RNG_NAME
        rank = self.distenv.world_rank # global rank
        epoch = 'last' if epoch == -1 else epoch
        ckpt_folder = os.path.join(self.config.result_path , CKPT_FOLDER.format(epoch))
        os.makedirs(ckpt_folder, exist_ok=True)
        xm.rendezvous("save_ckpt")
        # we gather a random state
        if master_only and not self.distenv.master:
            # still save rng
            rng_state = {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'random': random.getstate(),
                'xm': xm.get_rng_state()
            }
            rng_path = os.path.join(ckpt_folder, RNG_NAME.format(rank))
            torch.save(rng_state, rng_path)
            return
        model_path = os.path.join(ckpt_folder, MODEL_NAME.format(rank))
        opt_path = os.path.join(ckpt_folder, OPT_NAME.format(rank))
        sch_path = os.path.join(ckpt_folder, SCH_NAME.format(rank))
        rng_path = os.path.join(ckpt_folder, RNG_NAME.format(rank))
        additional_path = os.path.join(ckpt_folder, ADDIONTIONAL_NAME.format(rank))
        model_weight = self.sync_and_to_cpu(self.model_woddp.state_dict())
        optimizer_weight = self.sync_and_to_cpu(optimizer.state_dict())
        scheduler_weight = self.sync_and_to_cpu(scheduler.state_dict())
        rng_state = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
            'xm': xm.get_rng_state()
        }
        torch.save(model_weight, model_path)
        torch.save(optimizer_weight, opt_path)
        torch.save(scheduler_weight, sch_path)
        torch.save(rng_state, rng_path)
        additional_attr_ckpt = {}
        for attr in additional_attr_to_save:
            target_module = getattr(self, attr).module if self.use_ddp else getattr(self, attr)
            additional_attr_ckpt[attr] = self.sync_and_to_cpu(target_module.state_dict())
        if len(additional_attr_ckpt) > 0:
            torch.save(additional_attr_ckpt, additional_path)
        if self.model_ema:
            ema_model_path = os.path.join(ckpt_folder, EMA_MODEL_NAME.format(rank))
            ema_model_weight = self.sync_and_to_cpu(self.model_ema_woddp.state_dict())
            torch.save(ema_model_weight, ema_model_path)
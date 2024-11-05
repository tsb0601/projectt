from ast import List, Set, Tuple
from git import Union
from platformdirs import user_documents_dir
from tqdm import tqdm
from .gaussian_diffusion import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def logclip(t, eps: float = 1e-20):
    return torch.log(t.clamp(min = eps))
class SimpleDiffusion(GaussianDiffusion):
    def __init__(
        self,
        *,
        size_ratio: float = 1.,
        schedule: ScheduleType.SHIFTED_CONSINE,
        pred_term = ModelMeanType.EPSILON,
        loss_type = LossType.WEIGHTED_MSE,
        diffusion_steps: int = 1000,
        used_timesteps: list[int] = None,
    ):
        assert isinstance(schedule, ScheduleType), f'Invalid schedule type {schedule}'
        assert isinstance(pred_term, ModelMeanType), f'Invalid pred_term type {pred_term}'
        assert isinstance(loss_type, LossType), f'Invalid loss_type {loss_type}'
        self.schedule = schedule
        self.pred_term = pred_term
        self.loss_type = loss_type
        self.size_ratio = size_ratio
        self.log_ratio = math.log(size_ratio)
        self.diffusion_steps = diffusion_steps
        used_timesteps = set(timestep + 1 for timestep in used_timesteps) if used_timesteps is not None else set(range(1, diffusion_steps + 1)) # default to all timesteps
        # sort the timesteps in ascending order
        self.used_timesteps = sorted(used_timesteps)
        
    def logsnr_t(self, t: Union[torch.Tensor, float], schedule: ScheduleType, log_min: float = -15, log_max: float=15) -> torch.Tensor:
        """
            return the log SNR at time t
            logSNR_t = - 2 log (tan(pi * t / 2)) # use consine schedule
            if shifted cosine, logSNR_t_shifted = logSNR_t + 2 self.log_ratio
        """
        logsnr_max = log_max + self.log_ratio
        logsnr_min = log_min + self.log_ratio
        t_min = math.atan(math.exp(logsnr_min / 2)) 
        t_max = math.atan(math.exp(logsnr_max / 2))
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
        t_boundary = t_min + (t_max - t_min) * t
        logsnr_t = -2 * torch.log(torch.tan(t_boundary))
        if schedule == ScheduleType.SHIFTED_CONSINE:
            logsnr_t += 2 * self.log_ratio
        return logsnr_t
    def q_sample(self, x_start, alpha_t, sigma_t, noise= None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return alpha_t * x_start + sigma_t * noise
        
    def training_losses(self, model: nn.Module, x_start: torch.Tensor, t:torch.Tensor = None, model_kwargs=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        if t is None:
            t = torch.rand(x_start.size(0), device=x_start.device) # sample t from uniform distribution
        else:
            assert t.dtype is not torch.long and t.dtype is not torch.int, f'Invalid t dtype {t.dtype}'
        
        logsnr_t = self.logsnr_t(t, self.schedule)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x_start.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x_start.device) 
        z_t = self.q_sample(x_start, alpha_t, sigma_t, noise)
        model_pred = model(z_t, logsnr_t, **model_kwargs)
        assert model_pred.size() == x_start.size(), f'Invalid model prediction size {model_pred.size()}, expected {x_start.size()}'
        if self.pred_term == ModelMeanType.EPSILON:
            eps_pred = model_pred
            target = noise
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')
        
        snr = torch.exp(logsnr_t) # see https://arxiv.org/pdf/2303.09556
        if self.pred_term == ModelMeanType.EPSILON:
            #weighted_t = torch.clamp(snr, max = 5) / snr
            #weighted_t = weighted_t.view(-1, 1, 1, 1)
            #weighted_t = torch.ones_like(weighted_t)
            bias = - int(math.log2(1 / self.size_ratio)) - 1
            sigmoid_weight_t = torch.sigmoid(-logsnr_t + bias)
            weighted_t = sigmoid_weight_t.view(-1, 1, 1, 1)
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')

        if self.loss_type == LossType.WEIGHTED_MSE:
            loss = mean_flat(weighted_t * (eps_pred - target) ** 2)
        elif self.loss_type == LossType.MSE:
            loss = mean_flat((eps_pred - target) ** 2)
        else:
            raise NotImplementedError(f'Invalid loss type {self.loss_type}')
        terms = {
            'mse': loss,
            'loss': loss,
        }
        return terms
    def ddim_sample(self, model, x_t, t: float, s:float, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0):
        logsnr_t = self.logsnr_t(t, self.schedule).to(x_t.device)
        logsnr_s = self.logsnr_t(s, self.schedule).to(x_t.device)
        #print(f'logsnr_t: {logsnr_t}, logsnr_s: {logsnr_s}')
        model_pred = model(x_t, logsnr_t, **model_kwargs)
        c = torch.exp(logsnr_t - logsnr_s).view(-1, 1, 1, 1).to(x_t.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x_t.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x_t.device)
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s)).view(-1, 1, 1, 1).to(x_t.device)
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s)).view(-1, 1, 1, 1).to(x_t.device)
        if self.pred_term == ModelMeanType.EPSILON:
            x_pred = (x_t - sigma_t * model_pred) / alpha_t
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')
        if clip_denoised:
            x_pred = x_pred.clamp(-1, 1)        
        # for mu, variance, see https://arxiv.org/pdf/2410.19324
        mu = alpha_s * (c* x_t  + (1-c) * x_pred)
        gamma = .3 
        alpha_ts = alpha_t / alpha_s # \alpha_{t\mid s}
        sigma_ts_sq = sigma_t **2 - alpha_ts * sigma_s **2 # \sigma_{t\mid s}^2
        sigma_t2s_sq = 1/ (1/ sigma_s**2 + alpha_ts **2 / sigma_ts_sq) # \sigma_{t\to s}^2
        log_sigma_t2s= torch.log(sigma_t2s_sq) 
        log_sigma_ts = torch.log(sigma_ts_sq) 
        logvar = gamma * log_sigma_t2s + (1 - gamma) * log_sigma_ts
        variance = torch.exp(logvar) # \sigma_{t\to s}^\gamma \sigma_{t\mid s}^{1-\gamma}
        return mu, torch.zeros_like(variance)
    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0):
        if noise is None:
            noise = torch.randn(shape, device=device)
        x = noise.to(device)
        progess_bar = tqdm(reversed(range(1, len(self.used_timesteps)))) if progress else reversed(range(1, len(self.used_timesteps)))
        for i in progess_bar:
            #print(f'Processing timestep {i}:{self.used_timesteps[i]} to timestep {i - 1}:{self.used_timesteps[i - 1]}')
            u_t = self.used_timesteps[i] / self.diffusion_steps # current t
            u_s = self.used_timesteps[i - 1] / self.diffusion_steps # next t
            u_t = torch.tensor(u_t).to(device).repeat(x.size(0)) # repeat for batch size
            u_s = torch.tensor(u_s).to(device).repeat(x.size(0))
            z_mu, z_var = self.ddim_sample(model, x, u_t, u_s, clip_denoised, denoised_fn, cond_fn, model_kwargs, eta)
            x = z_mu + torch.randn_like(z_mu) * torch.sqrt(z_var)
            xm.mark_step()
        t_lowest = self.used_timesteps[0] / self.diffusion_steps
        t_lowest = torch.tensor(t_lowest).to(device).repeat(x.size(0))
        logsnr_t_lowest = self.logsnr_t(t_lowest, self.schedule).to(device)
        alpha_lowest = torch.sqrt(torch.sigmoid(logsnr_t_lowest)).view(-1, 1, 1, 1).to(device)
        sigma_lowest = torch.sqrt(torch.sigmoid(-logsnr_t_lowest)).view(-1, 1, 1, 1).to(device)
        model_pred = model(x, logsnr_t_lowest, **model_kwargs)
        if self.pred_term == ModelMeanType.EPSILON:
            x_pred = (x - sigma_lowest * model_pred) / alpha_lowest
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')
        if clip_denoised:
            x_pred = x_pred.clamp(-1, 1)
        return x_pred
        
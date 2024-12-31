import torch
from matplotlib import pyplot as plt
import numpy as np
import math
def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name in ["quad", "linear", "warmup10", "warmup50", "const", "jsd"]:
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
def get_beta_alpha(mode):
    num_diffusion_timesteps = 1000
    if mode == "linear":
        scale = 1000 / num_diffusion_timesteps
        betas = get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif mode == "squaredcos_cap_v2":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    
    #betas = get_named_beta_schedule(mode, num_diffusion_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    return betas, alphas, alphas_cumprod, sqrt_one_minus_alphas_cumprod
if __name__ == '__main__':
    num_diffusion_timesteps = 1000
    betas_s = []
    alphas_s = []
    #modes = ["quad", "linear", "warmup10", "warmup50", "const", "jsd", "squaredcos_cap_v2"]
    modes = ["linear","squaredcos_cap_v2"]
    for mode in modes:
        betas, alphas, alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_beta_alpha(mode)
        betas_s.append(betas)
        alphas_s.append(sqrt_one_minus_alphas_cumprod)
    for i in range(len(modes)):
        plt.plot(betas_s[i], label=modes[i])
    plt.legend()
    plt.savefig(f'./visuals/betas_{mode}.png')
    plt.clf()
    for i in range(len(modes)):
        np_alpha = np.array(alphas_s[i])
        log_alpha = np.log(np_alpha)
        plt.plot(log_alpha, label=modes[i])
    #print(alphas_s[0][::50], sep=', ')
    for k in range(0, len(alphas_s[0]), 50):
        print(alphas_s[0][k], end=', ')
    plt.legend()
    plt.savefig(f'./visuals/alphas_{mode}.png')
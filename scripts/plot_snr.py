from typing import Union
import torch
import math
def logsnr_t(t: Union[torch.Tensor, float], schedule: bool = False, log_min: float = -15, log_max: float=15) -> torch.Tensor:
    """
        return the log SNR at time t
        logSNR_t = - 2 log (tan(pi * t / 2)) # use consine schedule
        if shifted cosine, logSNR_t_shifted = logSNR_t + 2 self.log_ratio
    """
    log_ratio = math.log(64/256)
    logsnr_max = log_max
    logsnr_min = log_min
    t_min = math.atan(math.exp(-logsnr_max / 2)) 
    t_max = math.atan(math.exp(-logsnr_min / 2))
    t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    t_boundary = t_min + (t_max - t_min) * t
    logsnr_t = -2 * torch.log(torch.tan(t_boundary))
    #print(logsnr_t)
    #logsnr_t = -2 * torch.log(torch.tan(t_boundary))
    if schedule:
        logsnr_t += 2 * log_ratio
    return logsnr_t

def plot_logsnr(save_path: str = './visuals/logsnr.png'):
    import matplotlib.pyplot as plt
    t = torch.linspace(0, 1, 100)
    #print(t)
    logsnr = logsnr_t(t, schedule=False)
    logsnr_shifted = logsnr_t(t, schedule=True)
    plt.plot(t, logsnr, label='logSNR')
    plt.plot(t, logsnr_shifted, label='logSNR shifted')
    plt.xlabel('t')
    plt.ylabel('logSNR')
    plt.legend()
    plt.savefig(save_path)
    alpha_t = torch.sqrt(torch.sigmoid(logsnr)).view(-1)
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr)).view(-1)
    alpha_t_shifted = torch.sqrt(torch.sigmoid(logsnr_shifted)).view(-1)
    sigma_t_shifted = torch.sqrt(torch.sigmoid(-logsnr_shifted)).view(-1)
    # also plot them
    plt.clf()
    plt.plot(t, alpha_t, label='alpha_t')
    plt.plot(t, sigma_t, label='sigma_t')
    plt.plot(t, alpha_t_shifted, label='alpha_t shifted')
    plt.plot(t, sigma_t_shifted, label='sigma_t shifted')
    plt.xlabel('t')
    plt.ylabel('alpha_t, sigma_t')
    plt.legend()
    plt.savefig(save_path.replace('.png', '_alpha_sigma.png'))
    print(f'LogSNR saved to {save_path}')
if __name__ == '__main__':
    plot_logsnr()
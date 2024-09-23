from rqvae.optimizer.scheduler import GradualWarmup, Scheduler, LinearLR, CosineAnnealingLR
import math
import torch.nn as nn
import torch
class dummy(nn.Module):
    def __init__(self):
        super(dummy, self).__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

def create_scheduler(optimizer):
    init_lr = 1.0e-3
    multiplier = 1
    warmup_steps = 10
    buffer_steps = 0
    final_steps = 50
    min_lr = 1e-4
    mode = 'cosine'
    start_from_zero = True
    decay_end_epoch = 30
    steps_per_epoch = 1
    decay_steps = decay_end_epoch * steps_per_epoch - warmup_steps - buffer_steps
    
    if warmup_steps > 0.0:
        warmup = GradualWarmup(
            optimizer,
            steps=warmup_steps,
            buffer_steps=buffer_steps,
            multiplier=multiplier,
            start_from_zero=start_from_zero
        )
    else:
        warmup = None
    start_factor = multiplier
    decay_mode = 'cosine'
    if decay_mode == 'linear':
        end_factor = min_lr / init_lr
        print(f'end_factor: {end_factor}, last_lr : {init_lr * end_factor} decay_steps: {decay_steps}')
        scheduler = LinearLR(optimizer, start_factor, end_factor, decay_steps , last_epoch=-1)
    elif decay_mode == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max = decay_steps, eta_min=min_lr, last_epoch= -1
        )
    else:
        raise NotImplementedError(f'{decay_mode} is not a valid decay policy')
    scheduler = Scheduler(warmup_scheduler=warmup, after_scheduler=scheduler, decay_steps=decay_steps)

    return scheduler

def main():
    model = dummy()
    optim = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = create_scheduler(optim)
    for i in range(50):
        scheduler.step()
        print(f'{i} lr: {optim.param_groups[0]["lr"]:.4e}')
        
if __name__ == '__main__':
    main()
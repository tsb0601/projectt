from typing import Iterable
import torch
class norm_tracker:
    def __init__(self, parameters: Iterable, p: int = 2, max_norm: float = 1.0):
        self.p = p
        self.grad = []
        self.max_norm = max_norm
        self.parameters = [p for p in parameters if p.requires_grad]
        self.param_lens = [p.numel() for p in self.parameters]
    @torch.no_grad()
    def __call__(self):
        raise NotImplementedError("This class is not callable. Use the methods instead.")
        avg_grad = 0.
        total_param_len = 0
        for p in self.parameters:
            if p.grad is not None:
                avg_grad += p.grad.detach()**(self.p).sum() # lp-norm
                total_param_len += p.numel()            
        avg_grad = avg_grad**(1/self.p)
        grad_norm = avg_grad / total_param_len
        return grad_norm
    @torch.no_grad()
    def clip_norm(self):
        current_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_norm)
        return current_grad_norm
    
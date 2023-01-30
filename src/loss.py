import torch
from torch import nn

class MemLoss(nn.Module):
    def __init__(self, device) -> None:
        super(MemLoss, self).__init__()
        self.device = device
        
    def forward(self, x, x_T):
        m, _ = x.size()
        cosim = torch.matmul(x, x_T)/2
        id_mask = torch.eye(m).to(self.device)
        loss = torch.mean(torch.abs(cosim - id_mask))
        
        return loss
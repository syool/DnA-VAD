import torch
from torch import nn
import torch.nn.functional as F

import math


class Memory(nn.Module):
    def __init__(self, dim, num_item=100, k=10) -> None:
        super(Memory, self).__init__()
        mempool = nn.Parameter(torch.Tensor(num_item, dim))
        self.mempool = self.init_memory(mempool)
        self.dim = dim
        self.k = k
        
    def init_memory(self, memory):
        stdv = 1. / math.sqrt(memory.size(1))
        memory.data.uniform_(-stdv, stdv)

        return memory
    
    def disloss(self, x, x_T):
        pci = x.get_device()
        device = torch.device(f'cuda:{pci}')
        
        m, _ = x.size()
        cosim = torch.matmul(x, x_T)/2
        id_mask = torch.eye(m).to(device)
        loss = torch.mean(torch.abs(cosim*(1-id_mask)))
        
        return loss
    
    def memorize(self, input):
        shape = input.shape
        
        # == GET QUERIES ==
        if len(shape) == 4:
            input = input.permute(0, 2, 3, 1)
            
        input = input.contiguous()
        query = input.view(-1, shape[1])
        
        # == GET ATTENTION VECTORS ==
        att = F.linear(query, self.mempool)
        att = F.softmax(att, dim=1)
        
        # == Top-K SELECTION ==
        val, idx = torch.topk(att, k=self.k, dim=1)
        val = F.softmax(val, dim=1)
        att = torch.zeros_like(att).scatter_(1, idx, val)
        
        # == MEMORY SELECTION ==
        mempool_T = self.mempool.permute(1, 0)
        output = F.linear(att, mempool_T)
        
        loss = self.disloss(self.mempool, mempool_T)
        
        # == RECOVER DIMENSIONALITY ==
        if len(shape) == 4:
            output = output.view(shape[0], shape[2], shape[3], shape[1])
            output = output.permute(0, 3, 1, 2)
            
        return output, loss

    def forward(self, input):
        output, loss = self.memorize(input)
        
        return output, loss
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
    
    def dloss(self, x, x_T):
        pci = x.get_device()
        device = torch.device(f'cuda:{pci}')
        
        m, _ = x.size()
        cosim = torch.matmul(x, x_T)/2
        id_mask = torch.eye(m).to(device)
        # loss = torch.mean(torch.abs(cosim - id_mask))
        loss = torch.mean(torch.abs(cosim*(1-id_mask)))
        
        return loss
    
    def memorize(self, input1, input2):
        shape = input1.shape
        
        # == GET QUERIES ==
        if len(shape) == 4:
            input1 = input1.permute(0, 2, 3, 1)
            input2 = input2.permute(0, 2, 3, 1)
            
        input1 = input1.contiguous()
        query1 = input1.view(-1, shape[1])
        
        input2 = input2.contiguous()
        query2 = input2.view(-1, shape[1])
        
        # == GET ATTENTION VECTORS ==
        att1 = F.linear(query1, self.mempool)
        att1 = F.softmax(att1, dim=1)
        
        att2 = F.linear(query2, self.mempool)
        att2 = F.softmax(att2, dim=1)
        
        # == Top-K SELECTION ==
        val1, idx1 = torch.topk(att1, k=self.k, dim=1)
        val1 = F.softmax(val1, dim=1)
        att1 = torch.zeros_like(att1).scatter_(1, idx1, val1)
        
        val2, idx2 = torch.topk(att2, k=self.k, dim=1)
        val2 = F.softmax(val2, dim=1)
        att2 = torch.zeros_like(att2).scatter_(1, idx2, val2)
        
        # == MEMORY SELECTION ==
        mempool_T = self.mempool.permute(1, 0)
        output1 = F.linear(att1, mempool_T)
        output2 = F.linear(att2, mempool_T)
        
        # loss = self.dloss(self.mempool, mempool_T)
        
        # == RECOVER DIMENSIONALITY ==
        if len(shape) == 4:
            output1 = output1.view(shape[0], shape[2], shape[3], shape[1])
            output1 = output1.permute(0, 3, 1, 2)
            output2 = output2.view(shape[0], shape[2], shape[3], shape[1])
            output2 = output2.permute(0, 3, 1, 2)
            
        return output1, output2

    def forward(self, input1, input2):
        output1, output2 = self.memorize(input1, input2)
        
        return (output1, output2)
import torch
from torch import nn
from modules import AppearanceEncoder, MotionEncoder,\
                    Decoder, Memory, Attention


class Model(nn.Module):
    def __init__(self, clip_length=None, dataset=None) -> None:
        super(Model, self).__init__()
        self.a_encoder = AppearanceEncoder(clip_length)
        self.m_encoder = MotionEncoder()
        self.decoder = Decoder()
        
        if dataset == 'ped2':
            topk = [16, 8, 4]
        elif dataset == 'avenue':
            topk = [10, 8, 6]
        elif dataset == 'shanghai':
            topk = [20, 16, 12]
        
        self.me1 = Memory(64, 200, topk[0])
        self.me2 = Memory(128, 200, topk[1])
        self.me3 = Memory(256, 200, topk[2])
        
        self.at1 = Attention(64)
        self.at2 = Attention(128)
        self.at3 = Attention(256)
        self.atz = Attention(512)

    def forward(self, frame, flow):
        z_frame, s_frame = self.a_encoder(frame)
        z_flow, s_flow = self.m_encoder(flow)
        
        integrated = (self.at1(s_frame[0], s_flow[0]),
                      self.at2(s_frame[1], s_flow[1]),
                      self.at3(s_frame[2], s_flow[2]))
        z = self.atz(z_frame, z_flow)
        
        wired1, dloss1 = self.me1(integrated[0])
        wired2, dloss2 = self.me2(integrated[1])
        wired3, dloss3 = self.me3(integrated[2])
        
        wired = (wired1, wired2, wired3)
        output = self.decoder(z, wired)
        dloss = 0.1 * torch.mean(dloss1 + dloss2 + dloss3)

        return output, dloss


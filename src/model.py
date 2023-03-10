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
            topk = [5, 4, 3] # 5, 4, 3
        elif dataset == 'avenue':
            topk = [10, 8, 6] # 10, 8, 6
        elif dataset == 'shanghai':
            topk = [16, 8, 4]
        
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
        
        wired1 = self.me1(s_frame[0], s_flow[0])
        wired2 = self.me2(s_frame[1], s_flow[1])
        wired3 = self.me3(s_frame[2], s_flow[2])
        
        skip = (self.at1(wired1[0], wired1[1]),
                self.at2(wired2[0], wired2[1]),
                self.at3(wired3[0], wired3[1]),)
        z = self.atz(z_frame, z_flow)
        
        output = self.decoder(z, skip)
        # dloss = 0.01 * torch.mean(dloss1 + dloss2 + dloss3)

        return output


import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    ''' Appearance-Motion Feature Attention '''
    def __init__(self, c) -> None:
        super(Attention, self).__init__()
        def conv1x1(c):
            return nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(c)
            )
        def conv1x1_(c):
            return nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        self.g = conv1x1(c)
        self.f = conv1x1(c)
        self.v = conv1x1_(c)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, z_frame, z_flow):
        g = self.g(z_flow)
        x = self.f(z_frame)
        gx = self.relu(g+x)
        z = self.v(gx) # TODO: print coefficient maps

        return z * z_frame



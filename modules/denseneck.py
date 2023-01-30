import torch
from torch import nn

from .memory import Memory


class Denseneck(nn.Module):
    def __init__(self) -> None:
        super(Denseneck, self).__init__()
        
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(c_out),
                nn.ReLU(),
                
                nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
            
        def block_(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
            )
            
        self.tunnel1 = block(512, 128)
        self.tunnel2 = block(512+128, 128)
        self.tunnel3 = block(512+(128*2), 128)
        self.tunnel4 = block(512+(128*3), 128)
        self.tunnel5 = block_(512+(128*4), 512)

    def forward(self, z):
        z1 = self.tunnel1(z)
        cat1 = torch.cat((z, z1), dim=1)
        
        z2 = self.tunnel2(cat1)
        cat2 = torch.cat((cat1, z2), dim=1)
        
        z3 = self.tunnel3(cat2)
        cat3 = torch.cat((cat2, z3), dim=1)
        
        z4 = self.tunnel4(cat3)
        cat4 = torch.cat((cat3, z4), dim=1)
        
        z = self.tunnel5(cat4)

        return z
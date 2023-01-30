import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=1, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                
                nn.Conv2d(c_out, c_out, 3, stride=1, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
        def block_(c_in, c_out):
            c_mid = int(c_in/2)
            return nn.Sequential(
                nn.Conv2d(c_in, c_mid, 3, stride=1, padding=1),
                nn.BatchNorm2d(c_mid),
                nn.ReLU(),
                
                nn.Conv2d(c_mid, c_mid, 3, stride=1, padding=1),
                nn.BatchNorm2d(c_mid),
                nn.ReLU(),
                
                nn.Conv2d(c_mid, c_out, 3, stride=1, padding=1),
                nn.Tanh()
            )
        def upsample(c_in, c_out):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
            
        self.up4 = upsample(512, 256)
        
        self.conv3 = block(512, 256)
        self.up3 = upsample(256, 128)
        
        self.conv2 = block(256, 128)
        self.up2 = upsample(128, 64)
        
        self.conv1 = block_(128, 3)

    def forward(self, x, skip):
        u4 = self.up4(x)
        cat3 = torch.cat((skip[2], u4), dim=1)
        
        c3 = self.conv3(cat3)
        u3 = self.up3(c3)
        cat2 = torch.cat((skip[1], u3), dim=1)
        
        c2 = self.conv2(cat2)
        u2 = self.up2(c2)
        cat1 = torch.cat((skip[0], u2), dim=1)
        
        output = self.conv1(cat1)

        return output
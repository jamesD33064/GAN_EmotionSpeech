import torch.nn as nn
import Layer

class Generator(nn.Module):
    def __init__(self, latents):
        super(Generator, self).__init__()
        
        self.layer1= nn.Sequential(
            # input is random_Z,  state size. latents x 1 x 1 
            # going into a convolution
            Layer.TCBR(latents, 256, 4, 2, 1),  # state size. 256 x 2 x 2
            Layer.CBR(256, 128, 3, 1)
        )
        
        self.layer2= nn.Sequential(
            Layer.TCBR(128, 256, 4, 1, 0), # state size. 256 x 3 x 3
            Layer.TCBR(256, 256, 4, 2, 1), # state size. 256 x 6 x 6
            
        )
        self.layer3= nn.Sequential(
            Layer.TCBR(256, 128, 4, 1, 0), # state size. 256 x 7 x 7
            Layer.TCBR(128, 128, 4, 2, 1),  # state size. 256 x 14 x 14
            Layer.CBR(128, 128, 3, 1)
            # state size. 256 x 6 x 6

        )
        self.layer4= nn.Sequential(
            Layer.TCBR(128, 64, 4, 2, 1), # state size. 64 x 28 x 28
            Layer.CBR(64, 64, 3, 1),
            Layer.CBR(64, 64, 3, 1),
            nn.Conv2d(64, 1, 3, 1, 1), # state size. 1 x 28 x 28
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
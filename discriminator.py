import torch.nn as nn
import Layer

class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            Layer.CBLR(1, 32, 3, 2), # b*32*14*14
            Layer.CBLR(32, 64, 3, 1), # b*64*14*14
            Layer.CBLR(64, 128, 3, 2), # b*128*7*7
            Layer.CBLR(128, 128, 3, 2), # b*32*3*3
            Layer.CBLR(128, 64, 3, 2), # b*32*1*1
        )        
        self.fc = nn.Linear(64,2)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        ft = x
        output = self.fc(x)
        return output
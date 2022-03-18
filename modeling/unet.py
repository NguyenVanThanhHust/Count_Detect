import numpy as np
import torch.nn.functional as F
from torch import nn

from layers.basic_block import Encoder, Decoder

class Crack_Unet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), 
                       dec_chs=(1024, 512, 256, 128, 64), 
                       num_class=1, 
                       retain_dim=True, 
                       out_sz=(448,448)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out
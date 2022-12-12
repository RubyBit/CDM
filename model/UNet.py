import math
from inspect import isfunction
from functools import partial

#%matplotlib inline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
#from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

#https://github.com/g2archie/UNet-MRI-Reconstruction
#https://amaarora.github.io/2020/09/13/unet.html#understanding-input-and-output-shapes-in-u-net
# This implementation includes the added padding to prevent chnage of dimensions to produce image of same dimensions as input to reconstruct the image
# The dimensions were adjusted to be compatible with the dimensions of the time embedding of the MNIST dataset
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
class Encoder(nn.Module):
    def __init__(self, chs=(1,32,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 32)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
class UNet(nn.Module):
  def __init__(self, enc_chs=(1,32,64,128,256), dec_chs=(256, 128, 64, 32), num_class=1, retain_dim=False, out_sz=(572,572)):
      super().__init__()
      self.encoder     = Encoder(enc_chs)
      self.decoder     = Decoder(dec_chs)
      self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
      self.retain_dim  = retain_dim

  def forward(self, x):
      enc_ftrs = self.encoder(x)
      out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
      out      = self.head(out)
      if self.retain_dim:
          out = F.interpolate(out, out_sz)
      return out

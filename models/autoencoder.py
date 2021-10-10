# -*- coding: utf-8 -*-

import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
class AutoEncoder(nn.Module):
    def __init__(self, in_channel):
        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel

        self.encoder = nn.Sequential(nn.Linear(self.in_channel,512),
                                     nn.LayerNorm(512),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(512,256),
                                     nn.LayerNorm(256),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(256,128),
                                     nn.LayerNorm(128),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(128,2)
                                    )

        self.decoder = nn.Sequential(nn.Linear(2,128),
                                     nn.LayerNorm(128),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(128,256),
                                     nn.LayerNorm(256),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(256,512),
                                     nn.LayerNorm(512),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(512,self.in_channel),
                                    )
        
        self.apply(weights_init)
    
    def _encode(self, x):
        return self.encoder(x)
    
    def _decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)
        return x_recon
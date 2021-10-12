# -*- coding: utf-8 -*-

import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
class AutoEncoder(nn.Module):
    def __init__(self, in_channel:int, 
                 hidden_layer_sizes=(512,256,128)):
        
        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.hls = hidden_layer_sizes
        
        self.encoder = nn.Sequential(nn.Linear(self.in_channel,self.hls[0]),
                                     nn.LayerNorm(self.hls[0]),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(self.hls[0],self.hls[1]),
                                     nn.LayerNorm(self.hls[1]),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(self.hls[1],self.hls[2]),
                                     nn.LayerNorm(self.hls[2]),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(self.hls[2],2)
                                    )

        self.decoder = nn.Sequential(nn.Linear(2,self.hls[2]),
                                     nn.LayerNorm(self.hls[2]),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(self.hls[2],self.hls[1]),
                                     nn.LayerNorm(self.hls[1]),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(self.hls[1],self.hls[0]),
                                     nn.LayerNorm(self.hls[0]),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(self.hls[0],self.in_channel),
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
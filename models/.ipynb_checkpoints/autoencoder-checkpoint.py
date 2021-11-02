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

        def building_block(in_channel, out_channel):
            return [nn.Linear(in_channel,out_channel),
                    nn.LayerNorm(out_channel),
                    nn.LeakyReLU(0.02)]

        encoder = building_block(self.in_channel, self.hls[0])
        for i in range(len(self.hls)-1):
            encoder += building_block(self.hls[i], self.hls[i+1])
        encoder += [nn.Linear(self.hls[-1],2)]

        decoder = building_block(2, self.hls[-1])
        for i in range(len(self.hls)-1,0,-1):
            decoder += building_block(self.hls[i], self.hls[i-1])
        decoder += [nn.Linear(self.hls[0],self.in_channel),
                    nn.Softmax()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        self.apply(weights_init)
    
    def _encode(self, x):
        return self.encoder(x)
    
    def _decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)
        return x_recon
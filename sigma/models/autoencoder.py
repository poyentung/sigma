import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channel: int, hidden_layer_sizes=(512, 256, 128)):

        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.hls = hidden_layer_sizes

        def building_block(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

        encoder = building_block(self.in_channel, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += building_block(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], 2)]

        decoder = building_block(2, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += building_block(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.in_channel), nn.Softmax()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)
        return x_recon


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_channel: int, hidden_layer_sizes=(512, 256, 128)):

        super(VariationalAutoEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channel = in_channel
        self.hls = hidden_layer_sizes

        def building_block(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

        encoder = building_block(self.in_channel, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += building_block(self.hls[i], self.hls[i + 1])

        decoder = building_block(2, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += building_block(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.in_channel), nn.Softmax()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        self.logvar = nn.Linear(self.hls[-1], 2)
        self.mu = nn.Linear(self.hls[-1], 2)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _encode(self, x):
        encoder_out = self.encoder(x)
        return self.mu(encoder_out)

    def _decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoder_out = self.encoder(x)
        mu = self.mu(encoder_out)
        logvar = self.logvar(encoder_out)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        return mu, logvar, z, x_recon

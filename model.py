import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim, ngf):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        net = []

        channels = [z_dim, ngf*8, ngf*4, ngf*2, ngf]
        channels_out = [ngf*8, ngf*4, ngf*2, ngf, 3]
        active = ["relu", "relu", "relu", "relu", "tanh"]
        stride = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]
        for i in range(len(channels)):
            net.append(nn.ConvTranspose2d(channels[i], channels_out[i], 4, stride[i], padding[i], bias=False))
            if active[i] == "relu":
                net.append(nn.GroupNorm(32, channels_out[i]))
                net.append(nn.ReLU(True))
            elif active[i] == "tanh":
                net.append(nn.Tanh())

        self.generator = nn.Sequential(*net)
        self.weight_init()

    def weight_init(self):
        for m in self.generator.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

            elif isinstance(m, nn.GroupNorm):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.generator(x)
    

class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        net = []

        channels = [3, ndf, ndf*2, ndf*4, ndf*8]
        channels_out = [ndf, ndf*2, ndf*4, ndf*8, 1]
        active = ["lrelu", "lrelu", "lrelu", "lrelu", "sigmoid"]
        stride = [2, 2, 2, 2, 1]
        padding = [1, 1, 1, 1, 0]
        for i in range(len(channels)):
            net.append(nn.Conv2d(channels[i], channels_out[i], 4, stride[i], padding[i], bias=False))

            if i == 0:
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "lrelu":
                net.append(nn.GroupNorm(32, channels_out[i]))
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "sigmoid":
                net.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*net)
        self.weight_init()

    def weight_init(self):
        for m in self.discriminator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

            elif isinstance(m, nn.GroupNorm):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        out = self.discriminator(x)
        return out.view(x.size(0), -1)
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, enc, dec, identifier):
        super(AutoEncoder, self).__init__()
        self.enc, self.dec = enc, dec
        self.identifier = identifier

    @property
    def device(self):
        return self.enc.block1[0][0].weight.device

    @property
    def first_row(self):
        return self.enc.block1[0][0].weight[0]


class GeneratorStacked(nn.Module):
    def __init__(self, nz, ngf, nc, output_size, device, dropout_p=0.):
        super(GeneratorStacked, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.output_size = output_size
        self.device = device
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(nz, 256, 4, 1, 1, bias=False).to(device),
            nn.BatchNorm2d(ngf * 8).to(device),
            nn.LeakyReLU(0.3),
            self.dropout)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False).to(device),
            nn.BatchNorm2d(ngf * 4).to(device),
            nn.LeakyReLU(0.3),
            self.dropout)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False).to(device),
            nn.BatchNorm2d(ngf * 2).to(device),
            nn.LeakyReLU(0.3),
            self.dropout)
        if output_size == 16:
            self.block4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True).to(device),
                # nn.BatchNorm2d(32).to(device),
                # nn.LeakyReLU(0.3),
                # self.dropout)
                nn.Tanh())
        elif output_size == 28:
            self.block4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 2, bias=False).to(device),
                nn.BatchNorm2d(32).to(device),
                nn.LeakyReLU(0.3),
                self.dropout)
            self.block5 = nn.Sequential(
                nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=True).to(device),
                nn.Tanh()
            )
        elif output_size in [32, 128]:
            self.block4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False).to(device),
                nn.BatchNorm2d(32).to(device),
                nn.LeakyReLU(0.3),
                self.dropout)
            if output_size == 32:
                self.block5 = nn.Sequential(
                    nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=True).to(device),
                    nn.Tanh()
                )
            elif output_size == 128:
                self.block5 = nn.Sequential(
                    nn.ConvTranspose2d(32, 32, 4, 4, 0, bias=False).to(device),
                    nn.BatchNorm2d(32).to(device),
                    nn.LeakyReLU(0.3),
                    self.dropout)
                self.block6 = nn.Sequential(
                    nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=True).to(device),
                    nn.Tanh()
                )

    def forward(self, inp):
        out = self.block1(inp)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        if self.output_size in [28, 32, 128]:
            out = self.block5(out)
        if self.output_size == 128:
            out = self.block6(out)
        return out


class EncoderStacked(nn.Module):
    def __init__(self, block1, block2):
        super(EncoderStacked, self).__init__()
        self.block1, self.block2 = block1, block2

    def forward(self, inp):
        out1 = self.block1(inp)
        out2 = self.block2(out1)
        return out2

    @property
    def device(self):
        return self.enc.block1[0][0].weight.device

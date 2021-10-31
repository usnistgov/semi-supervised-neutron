from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, ngf, ngf2):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.ngf2 = ngf2
        self.main = nn.Sequential(
            nn.ConvTranspose1d(1, self.ngf, 17, 2, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 15, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 14, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 18, 3, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf2, 13, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 11, 4,bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),            
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 9,1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),            
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 9,1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),   
            nn.ConvTranspose1d( self.ngf2, 1, 7, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, noise):
        return self.main(noise)
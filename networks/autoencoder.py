import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np

class encoder_(nn.Module):
    def __init__(self, cfg):
        super(encoder_, self).__init__()
       
       
        self.nef = cfg.nef #182
        self.nc = cfg.nc
        self.ndf = cfg.ndf
       
        self.ae_dims = cfg.nef
        self.e_ch_dims = 42 
        self.d_ch_dims = 21
        self.e_dims = 126
        self.resolution = 128
        self.lowest_dense_res = 128 // 16
        self.dims = self.nc * self.d_ch_dims

        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.encoder1 = nn.Sequential(
        #3*128*128
        nn.Conv2d(self.nc, self.nef, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        #128*64*64
        nn.Conv2d(self.nef, self.nef*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*2),
        nn.LeakyReLU(0.2, inplace=True),
        #256*32*32
        nn.Conv2d(self.nef*2, self.nef*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(self.nef*4, self.nef*8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*8),
        nn.LeakyReLU(0.2, inplace=True)
        #512*16*16
        )
        self.encoder2 = nn.Sequential(
        nn.Conv2d(self.nef*8, self.nef*4, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*4),
        nn.LeakyReLU(0.2, inplace=True),
        #1024*8*8
        nn.Conv2d(self.nef*4, self.nef*8, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*8),
        nn.LeakyReLU(0.2, inplace=True),
        #2048*4*4
        )
        self.encoder3 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024*8*8, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 8*8*512),
        nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, x):
        x = self.encoder1(x)
        residual1 = x
        x = self.encoder2(x)
        x = self.relu1(x+residual1)
        x = self.encoder3(x)
        x = x.view(-1,512,8,8)
        return x


class decoder_(nn.Module):
    def __init__(self, cfg):
        super(decoder_, self).__init__()
 
        self.nef = cfg.nef #182
        self.nc = cfg.nc
        self.ndf = cfg.ndf
       
        self.ae_dims = cfg.nef
        self.e_ch_dims = 42 
        self.d_ch_dims = 21
        self.e_dims = 126
        self.resolution = 128
        self.lowest_dense_res = 128 // 16
        self.dims = self.nc * self.d_ch_dims



        self.decoder1  = nn.Sequential(
        nn.ConvTranspose2d(self.nef*4, self.nef*4, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*4),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef*4, self.nef*4, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*4),
        nn.LeakyReLU(0.1, inplace=True)
        )
        self.decoder2 = nn.Sequential(
        nn.ConvTranspose2d(self.nef*4, self.nef*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*2),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef*2, self.nef*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*2),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef*2, self.nef, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef, self.nc, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        x = self.decoder1(x)
        x = self.decoder2(x+residual)
        return x

class autoencoder(nn.Module):
    def __init__(self, cfg, netE, netD):
        super(autoencoder, self).__init__()

        self.nef = cfg.nef #182
        self.nc = cfg.nc
        self.ndf = cfg.ndf
        
        self.enc = netE
        self.dec = netD

    def forward(self, x):
        x = self.enc(x)
        #print("size:", x.size())
        x = self.dec(x)
        return x

'''


class autoencoder(nn.Module):
    def __init__(self, cfg):
        super(autoencoder, self).__init__()

        self.nef = cfg.nef #182
        self.nc = cfg.nc
        self.ndf = cfg.ndf
       
        self.ae_dims = cfg.nef
        self.e_ch_dims = 42 
        self.d_ch_dims = 21
        self.e_dims = 126
        self.resolution = 128
        self.lowest_dense_res = 128 // 16
        self.dims = self.nc * self.d_ch_dims

 
        self.encoder = nn.Sequential(
        #3*128*128
        nn.Conv2d(self.nc, self.nef, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        #32*32*128
        nn.Conv2d(self.nef, self.nef*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*2),
        nn.LeakyReLU(0.2, inplace=True),
        #8*8*256
        nn.Conv2d(self.nef*2, self.nef*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*4),
        nn.LeakyReLU(0.2, inplace=True),
        #16*16*512
        nn.Conv2d(self.nef*4, self.nef*8, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*8),
        nn.LeakyReLU(0.2, inplace=True),
        #8*8*1024
        nn.Flatten(),
        nn.Linear(1024*16*16, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 8*8*512),
        nn.LeakyReLU(0.2, inplace=True)
        )


        self.decoder  = nn.Sequential(
        #nn.ConvTranspose2d(self.nef*4, self.nef*4, kernel_size=4, stride=2, padding=1, bias=False),
        #nn.BatchNorm2d(self.nef*4),
        #nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef*4, self.nef*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*4),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef*4, self.nef*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.nef*2),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef*2, self.nef, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(self.nef, self.nc, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Sigmoid()
        )
    

    def forward(self, x):
        x = self.encoder(x)
        #print("size:", x.size())
        x = x.view(-1,512,8,8)
        #print("size:", x.size())
        x = self.decoder(x)
        return x

'''

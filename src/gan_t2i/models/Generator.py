import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger

from tqdm import tqdm
import os


class Generator(nn.Module):
    
    """ GAN Generator """

    def __init__(self, embedding_dim , p_emb_dim):
        super(Generator, self).__init__()
        
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100                                 # z
        self.emb_dim = embedding_dim                         # dimension of the embedding coming from CLIP 
        self.p_emb_dim = p_emb_dim
        self.latent_dim = self.noise_dim + self.p_emb_dim
        self.ngf = 64                                        # ngf stands for "number of generator features" in this contex
    
    
        # Method 4.1 
        self.projection = nn.Sequential(
        nn.Linear(in_features=self.emb_dim, out_features=self.p_emb_dim),          # first layer to trasfrom from emb_dim to p_emb_dim
        nn.BatchNorm1d(num_features=self.p_emb_dim),                               # Normalization layer
        nn.LeakyReLU(negative_slope=0.2, inplace=True)                             # Leaky ReLu
        )

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.Generator_net = nn.Sequential(
        nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),  # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        nn.BatchNorm2d(self.ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (num_channels) x 64 x 64
        )


    def forward(self, embed_vector, z):

        #print("Forward Generator")
        # TODO: To implement
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.Generator_net(latent_vector)
        return output
            
            
            
            
            
            
        
            
            
            


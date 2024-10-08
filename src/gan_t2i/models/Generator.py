import torch
import torch.nn as nn


class Generator(nn.Module):
    """GAN Generator"""

    def __init__(self, embedding_dim, p_emb_dim):
        super(Generator, self).__init__()

        self.num_channels = 3
        self.noise_dim = 100  # Noise dimension
        self.emb_dim = (
            embedding_dim  # dimension of the embedding coming from CLIP - 512
        )
        self.p_emb_dim = p_emb_dim  # dimensione di proiezione - 128
        self.latent_dim = self.noise_dim + self.p_emb_dim  # Latent space
        self.ngf = 64  # ngf stands for "number of generator features" in this contex

        # Method 4.1
        self.projection = nn.Sequential(
            nn.Linear(in_features=self.emb_dim, out_features=self.p_emb_dim),
            nn.BatchNorm1d(num_features=self.p_emb_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.Generator_net_64 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (num_channels) x 64 x 64
        )

        self.Generator_net_224 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # dim (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                self.ngf * 8,
                self.ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf // 2),
            nn.ReLU(True),
            # state size. (ngf/2) x 64 x 64
            nn.ConvTranspose2d(self.ngf // 2, self.ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf // 4),
            nn.ReLU(True),
            # state size. (ngf/4) x 128 x 128
            nn.ConvTranspose2d(
                self.ngf // 4,
                self.num_channels,
                kernel_size=6,
                stride=2,
                padding=18,
                bias=False,
            ),  # H_out  = (128-1) x 2 - 2 x 1 + 4  = 8
            nn.Tanh(),
            # state size. (num_channels) x 224 x 224
        )

    def forward(self, embed_vector, z):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.Generator_net_64(latent_vector)

        return output

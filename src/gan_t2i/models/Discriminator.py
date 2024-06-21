import torch
import torch.nn as nn

class Discriminator(nn.Module):
    
    """ GAN Discriminator """
    def __init__(self , embedding_dim , p_emb_dim):
        """
        Initialize the Discriminator

        Args:
            embedding_dim (int): Dimension of the embedding coming from CLIP
        """
        super(Discriminator, self).__init__()

        # Initialize the image and channel parameters
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = embedding_dim                   # dimension of the embedding coming from CLIP 
        self.projected_embed_dim = p_emb_dim             # dimension of the embedding to obtain
        self.B_dim = 128
        self.C_dim = 16
        self.ndf = 64                                    # new dimension of the features

        # Define the architecture of the discriminator
        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        ######## Concat_embed class , useful to concatenate 2 embeddings of different size ###################
        class Concat_embed(nn.Module):
            def __init__(self, embed_dim, projected_embed_dim):
                super(Concat_embed, self).__init__()
                self.projection = nn.Sequential(
                    nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
                    nn.BatchNorm1d(num_features=projected_embed_dim),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                    )

            def forward(self, inp, embed):
                
                projected_embed = self.projection(embed)
                #print("Dimensioni di inp:", inp.size())
                #print("Dimensioni di embed:", embed.size())
                #print("Dimensioni di projected_embed:", projected_embed.size())
                #replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
                # Reshape projected_embed to match inp
                projected_embed = projected_embed.unsqueeze(2).unsqueeze(3)
                replicated_embed  = projected_embed.expand(-1, -1, inp.size(2), inp.size(3))   # 16 , 64 , 4 ,4 
                
                #print("Dimensioni di inp:", inp.size())
                #print("Dimensioni di Replicated_embed:", replicated_embed.size())
                
                hidden_concat = torch.cat([inp, replicated_embed], 1)
                return hidden_concat
                
                    
        ###########################################

        # Initialize the projector
        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        # Define the architecture for the final layer
        self.netD_2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )	
        
    def forward(self, inp, embed):
            """
            Forward pass of the discriminator

            Args:
                inp (torch.Tensor): Input images
                embed (torch.Tensor): Embeddings coming from CLIP

            Returns:
                tuple: Output logits and intermediate features
            """
            x_intermediate = self.netD_1(inp)
            x = self.projector(x_intermediate, embed)
            x = self.netD_2(x)

            return x.view(-1, 1).squeeze(1) , x_intermediate

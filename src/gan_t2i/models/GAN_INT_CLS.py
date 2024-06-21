from torchinfo import summary
import torch.nn as nn
import torch


class Generator(nn.Module):
    
    def __init__(self, emb_dim, proj_emb_dim = 128, noise_dim = 100,  num_gen_features = 64):
        """ 
        # GAN-INT-CLS Generator
        The generator network for a GAN-INT-CLS model. It produces images of size 64x64x3
        starting from a latent space
        ## Args
        - emb_dim: The original size of the caption embedding
        - proj_emb_dim: The size of the projection of the caption embedding   
        - noise_dim: The original size of the noise vecotr z
        - num_gen_features: The number of generator features across the ConvT2D layers     
        """
        super(Generator, self).__init__()

        self.emb_dim = emb_dim                  # The original size of the caption embedding
        self.noise_dim = noise_dim              # The original size of the noise vecotr z
        self.proj_emb_dim = proj_emb_dim        # The size of the projection of the caption embedding
        self.latent_dim = noise_dim + proj_emb_dim      # The size of the latent space for the generator
        self.n_gf = num_gen_features            # The number of generator features


        # The projection network of the caption embedding
        self.proj_emb_net = nn.Sequential(
            nn.Linear(
                in_features=self.emb_dim,
                out_features=self.proj_emb_dim
            ),
            nn.BatchNorm1d(num_features=self.proj_emb_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


        # The generator network. Based on: https://github.com/aelnouby/Text-to-Image-Synthesis/blob/master/models/wgan_cls.py
        self.generator_net = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.n_gf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.n_gf * 8),
			nn.ReLU(True),
			# Current size => (n_gf * 8) x 4 x 4

			nn.ConvTranspose2d(self.n_gf * 8, self.n_gf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.n_gf * 4),
			nn.ReLU(True),
			# Current size => (n_gf * 4) x 8 x 8
			
            nn.ConvTranspose2d(self.n_gf * 4, self.n_gf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.n_gf * 2),
			nn.ReLU(True),
			# Current size => (n_gf * 2) x 16 x 16
			
            nn.ConvTranspose2d(self.n_gf * 2,self.n_gf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.n_gf),
			nn.ReLU(True),
			# Current size => (n_gf) x 32 x 32
			
            nn.ConvTranspose2d(self.n_gf, 3, 4, 2, 1, bias=False),
			nn.Tanh()
            # Output size => 3 x 64 x 64
        )



    def forward(self, emb_caption, z):
        # Project the caption embedding 
        emb_proj = self.proj_emb_net(emb_caption)

        # Concatenate the projected caption embedding and the noise vector z
        latent_vector = torch.cat([emb_proj, z], 1)

        # Add a third and fourth dimension to the latent vector to match the dimension of the generator network
        # ConvTranspose2D since it needs a tensor with 4 dimensions (batch_size, channels, height, width) 
        latent_vector = latent_vector.unsqueeze(2).unsqueeze(3)

        # Generate the image from the latent vector
        return self.generator_net(latent_vector)





class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):
        return x


class GAN_INT_CLS(nn.Module):
    
    def __init__(self, emb_network, emb_dim, proj_emb_dim = 128, noise_dim = 100, num_gen_features = 64):
        """ 
        # The Gan-INT-CLS model 
        The GAN-INT-CLS model. It produces images of size 64x64x3 starting from a caption
        ## Args
        - emb_network: The embedding network used to produce caption embeddings
        - embedding_size: The size of the caption embedding (size of the emb_network output)
        - proj_emb_dim: The size of the projection of the caption embedding
        - noise_dim: The original size of the noise vecotr z
        - num_gen_features: The number of generator features across the ConvT2D layers
        """
        super(GAN_INT_CLS, self).__init__()

        self.emb_dim = emb_dim
        self.proj_emb_dim = proj_emb_dim
        self.noise_dim = noise_dim
        self.num_gen_features = num_gen_features


        # The embedding network
        self.emb_net = emb_network

        # The generator network
        self.generator = Generator(
            emb_dim = emb_dim,
            proj_emb_dim = proj_emb_dim,
            noise_dim = noise_dim,
            num_gen_features = num_gen_features
        )

        # The discriminator network
        self.discriminator = Discriminator()
        

    def fit(self, train_dataloader, val_dataloader = None, device = "cuda" if torch.cuda.is_available() else "cpu", epochs = 600):
        #TODO: Finish to implement the fit function
        
        for epoch in range(epochs):
            
            for images, captions, _ in train_dataloader:
                
                images = images.to(device)
                captions = captions.to(device)
        
                self.generator(captions[0])
    

    def summary(self):
        """ 
        # Summary
        Prints the summary of the model 
        """

        # Generator summary
        summary(self.generator, [(1, self.emb_dim), (1, self.noise_dim)])

        # TODO: Discriminator summary

        # TODO: GAN-INT-CLS summary
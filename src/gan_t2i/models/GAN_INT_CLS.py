import torch.nn as nn
import torch

class Generator(nn.Module):
    
    def __init__(self, emb_dim = 1024, noise_dim = 100, proj_emb_dim = 128):

        self.emb_dim = emb_dim                  # The original size of the caption embedding
        self.noise_dim = noise_dim              # The original size of the noise vecotr z
        self.proj_emb_dim = proj_emb_dim        # The size of the projection of the caption embedding
        self.gen_latend_dim = noise_dim + proj_emb_dim      # The size of the latent space for the generator
        


        super(Generator, self).__init__()
        

        self.proj_emb = nn.Sequential(
            
        )


        self.generator_net = nn.Sequential(
        
        )




    def forward(self, x):
        return x


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        print("Creating GAN INT CLS Discriminator")

    def forward(self, x):
        return x


class GAN_INT_CLS(nn.Module):
    
    def __init__(self):
        super(GAN_INT_CLS, self).__init__()
                
        self.generator = Generator()
        self.discriminator = Discriminator()
        
        print("Creating GAN INT CLS Model")


    def fit(self, train_dataloader, val_dataloader = None, device = "cuda" if torch.cuda.is_available() else "cpu", epochs = 600):
        
        
        for epoch in range(epochs):
            
            for images, captions, _ in train_dataloader:
                
                images = images.to(device)
                captions = captions.to(device)
        
                self.generator()
        

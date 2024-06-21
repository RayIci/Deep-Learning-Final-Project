from gan_t2i.models.Generator import Generator
from gan_t2i.models.Discriminator import Discriminator
from torch.utils.data import DataLoader

import torch.nn as nn
import torch 
import yaml

class WGAN(nn.Module):
    
    def __init__(self, text_emb_model , embedding_size , p_emb_dim):
        
        super(WGAN, self).__init__()
        
        self.text_emb_model = text_emb_model
        
        # TODO : read PATH about the GAN Net
        #with open('config.yaml', 'r') as f:
        #    config = yaml.load(f)
            
        # TODO: Extract embedding size from CLIP
        # coming from outside
       

        # TODO: Create generator and discriminator network 
        self.generator = torch.nn.DataParallel( Generator(embedding_size , p_emb_dim ) )
        self.discriminator = torch.nn.DataParallel( Discriminator(embedding_size , p_emb_dim ) )
        
        
        # TODO: load checkpoints coming from already executed net
        
        
    def train(self, dataset):
        
        # TODO : read info related to the net 
        self.noise_dim = 100
        self.batch_size = 32
        self.num_workers = 4
        self.lr = 0.05
        self.beta1 = 0.5
        self.num_epochs = 5
        self.DITER = 5

        self.l1_coef = 50
        self.l2_coef = 100
        
        self.dataset = dataset
        
        # TODO: load the DataLoader with the dataset and the related parameters
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
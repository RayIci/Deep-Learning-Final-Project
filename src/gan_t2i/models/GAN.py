from gan_t2i.models.Generator import Generator
from gan_t2i.models.Discriminator import Discriminator
from torch.utils.data import DataLoader

import torch.nn as nn
import torch 
import yaml
import os
from torch.autograd import Variable
from tqdm import tqdm

# to delete
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import random

class WGAN(object):
    
    def __init__(self, text_emb_model , embedding_size , p_emb_dim , device=("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(WGAN, self).__init__()
        
        
        self.device = device
        self.text_emb_model = text_emb_model       # Model CLIP already trained
        self.embedding_size = embedding_size       # output dimension of the last layer of CLIP - 512
        self.p_emb_dim = p_emb_dim                 # dimension is which we want to project the data - 128

        self.noise_dim = 100                       # Dimension of the noise z 
        self.batch_size = 32                       # Dimension of the batch ( not used )
        self.num_workers = 4
        self.DITER = 5                             # DISCRIMINANT ITERATION
        self.l1_coef = 50
        self.l2_coef = 100
        self.lr = 0.05
        self.beta1 = 0.5
        
        
        # Create generator and discriminator network 
        self.generator = torch.nn.DataParallel( Generator(embedding_size , p_emb_dim ) )
        self.discriminator = torch.nn.DataParallel( Discriminator(embedding_size , p_emb_dim ) )
        
        # Set up Discriminator and Generator Optimizer Adam 
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            
        
       
    # TODO: load checkpoints coming from already executed net 
    @staticmethod
    def load(model_pt_filepath, device=("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Load a model from a checkpoint file.

        Args:
            model_pt_filepath (str): The path to the checkpoint file.
            device (str, optional): The device to use. Defaults to "cuda" if CUDA is available, otherwise "cpu".

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.

        Returns:
            tuple: A tuple containing the epoch number, total number of steps, training loss for the generator, 
            and training loss for the discriminator.

        """
        if not os.path.exists(model_pt_filepath):
            raise FileNotFoundError(f"Checkpoint path {model_pt_filepath} does not exist")
        
        checkpoint = torch.load(model_pt_filepath, map_location=device)
        text_emb_model = checkpoint["text_emb_model"]
        embedding_size = checkpoint["embedding_size"]
        p_emb_dim = checkpoint["p_emb_dim"]
        

        loaded_model = WGAN(text_emb_model,embedding_size,p_emb_dim)
        
        # loading generator and discriminator already trained
        loaded_model.generator.load_state_dict(checkpoint["gen_model_state_dict"])
        loaded_model.optimG.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        
        loaded_model.discriminator.load_state_dict(checkpoint["discr_model_state_dict"])
        loaded_model.optimD.load_state_dict(checkpoint["discr_optimizer_state_dict"])
        
        epoch = checkpoint["epoch"]
        
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}.")
        
        return loaded_model
        
    def fit(self, train_dataloader , val_dataloader , num_epochs , save_path ,starting_epoch = 0):
        
        # read info related to the net 
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs


        print("Training on device: ", self.device)
        
        total_steps = 0                             # Counter of total number of iterations
        STEPS_PER_EPOCH = len(train_dataloader)
    
        
        # Validate the checkpoint path or create it
        if save_path is not None and os.path.exists(save_path) and not os.path.isdir(save_path):
            raise ValueError(f"Checkpoint path {save_path} does not exist or isn't a directory.")
        elif save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)
        elif save_path is not None and os.path.exists(save_path) and os.path.isdir(save_path):
            for file in os.listdir(save_path):
                if file.endswith(".pt"):
                    print("Already exist .pt files related to some epochs , fit will overwrite from epoch {}".format(starting_epoch))
                    #raise ValueError(f"Checkpoint path {save_path} contains a checkpoint file, the checkpoint path must be empty ({file} remove it to continue).")
        
        # train WGAN
        one = torch.FloatTensor([1])
        mone = one * -1
        one = Variable(one).to(self.device)
        mone = Variable(mone).to(self.device)
        print("One : {} | Mone : {} ".format(one,mone))
        

        # TRAIN THE DISCRIMINATOR AND GENERATOR
        
        for epoch in range(self.num_epochs):
            print("epoch iter {} ".format(starting_epoch+epoch))
            
            # Tell the model that we are training
            self.discriminator.train()

            # Progress bar
            pbar = tqdm(total=STEPS_PER_EPOCH)
            running_loss_gen = 0.0
            running_loss_discr = 0.0
            
            for total_processed_batches, (images, captions, _) in enumerate(self.train_dataloader):
                images, captions = images.to(self.device), captions.to(self.device)
                
                # Encode captions using the text_emb_model
                captions_embeddings = self.text_emb_model.encode_text(captions).to(self.device)
                # image_embeddings = self.text_emb_model.encode_image(images) #.to(self.device)

                # Extract Real Image and related Caption
                right_images = Variable(images.float()).to(self.device)                    
                right_embeds = Variable(captions_embeddings.float()).to(self.device)       
                
                # Train the discriminator , parameter trainable 
                for p in self.discriminator.parameters():
                    p.requires_grad = True

                # ---------------------
                #  Train Discriminator
                # ---------------------
                for j in range(self.DITER):                             

                    
                    self.optimD.zero_grad()                          
                    
                    # Generate salt
                    z = torch.randn(images.size(0), self.noise_dim, 1, 1).to( self.device).float()
                    
                    
                    # Generate Fake image based on salt
                    # Detach  useful to not compute the gradient in the generator because we are training the discriminator
                    fake_images = self.generator(right_embeds, z).detach()          
                    wrong_images = Variable(fake_images.float())                    
                    
                    # Free up unused memory
                    torch.cuda.empty_cache()
                    
                    # Discriminate real image , real caption
                    d_real, _ = self.discriminator(right_images, right_embeds)  
                    
                    d_real_loss = torch.mean(d_real).unsqueeze(0)
                    # with Wasserstein GAN (WGAN) you want to MAXIMIZE the loss function (MONE)
                    (d_real_loss).backward(mone)                           

                    # Discriminate fake image , real caption
                    d_fake, _ = self.discriminator(wrong_images, right_embeds)  
                    d_fake_loss = torch.mean(d_fake).unsqueeze(0)
                    # with Wasserstein GAN (WGAN) you want to MINIMIZE the loss function (ONE)
                    d_fake_loss.backward(one)                              
                    
                    print("-- d_real_loss: {} , d_fake_loss: {}".format(d_real_loss,d_fake_loss))

                    # NOTE : Implement Gradient Penalty
                    
                    # compute of the loss function 
                    d_loss = d_real_loss - d_fake_loss
                    self.optimD.step()
                    total_steps += 1
                    running_loss_discr += d_loss.item()

                    
                    # Weight clipping
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    
                # Free up unused memory
                torch.cuda.empty_cache()
                
                
                # ---------------------
                #  Train Generator
                # ---------------------

                #train the generator 
                self.generator.train()
 
                self.optimG.zero_grad()
                
                # Generate salt 
                z = torch.randn(images.size(0), self.noise_dim, 1, 1).to(self.device).float() 
                
                # Generate Fake image
                fake_images = self.generator(right_embeds , z)                                 
                wrong_images = Variable(fake_images.float())
                
                # Discriminate fake image , real caption
                d_fake, _ = self.discriminator(wrong_images, right_embeds)                     
                g_loss = torch.mean(d_fake).unsqueeze(0)
                # we want to MINIMIZE the loss function 
                g_loss.backward(mone)                 
                g_loss = -g_loss
                self.optimG.step()
                
                # Update counters
                total_steps += 1
                running_loss_gen += g_loss.item()
                
                # Free up unused memory
                torch.cuda.empty_cache()
                
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Epoch [{starting_epoch+epoch+1}/{starting_epoch+num_epochs}] " + 
                                        f"Batch [{total_processed_batches+1}/{STEPS_PER_EPOCH}] \n " +
                                        f" => Loss Discriminator: {(running_loss_discr/(total_processed_batches+1)):.6f} \n" +
                                        f" => Loss Generator: {(running_loss_gen/(total_processed_batches+1)):.6f} \n " 
                                    )

            pbar.close()
            epoch_loss_gen = running_loss_gen / len(train_dataloader)
            epoch_loss_discr = running_loss_discr / len(train_dataloader)
            
            # Print the epoch summary after one epoch
            print(f"Epoch [{starting_epoch+epoch+1}/{num_epochs}] Summary: \n")
            print(f"\t=> Train Generator Loss: {epoch_loss_gen:.6f} \n ")
            print(f"\t=> Train Discriminator Loss: {epoch_loss_discr:.6f} \n ")
            print("")
                
            def Validation(val_dataloader):
                print("Validating...")
                self.generator.eval()
                self.discriminator.eval()

                val_loss_gen = 0.0
                val_loss_discr = 0.0

                # Disabling gradients during validation
                with torch.no_grad():  
                    for val_images, val_captions, _ in val_dataloader:
                        val_images, val_captions = val_images.to(self.device), val_captions.to(self.device)
                        
                        # Encode captions using the text embedding model
                        val_captions_embeddings = self.text_emb_model.encode_text(val_captions).to(self.device)

                        # Prepare validation images and captions embeddings
                        right_images = val_images.float().to(self.device)
                        right_embeds = val_captions_embeddings.float().to(self.device)

                        # Generate fake images
                        z = torch.randn(val_images.size(0), self.noise_dim, 1, 1).to(self.device).float()
                        fake_images = self.generator(right_embeds, z)

                        # Discriminate real images
                        d_real, _ = self.discriminator(right_images, right_embeds)
                        d_real_loss = torch.mean(d_real)

                        # Discriminate fake images
                        d_fake, _ = self.discriminator(fake_images, right_embeds)
                        d_fake_loss = torch.mean(d_fake)

                        # Compute losses for the generator and discriminator
                        d_loss = d_real_loss - d_fake_loss
                        g_loss = d_fake_loss


                        val_loss_discr += d_loss.item()
                        val_loss_gen += g_loss.item()

                val_loss_gen /= len(val_dataloader)
                val_loss_discr /= len(val_dataloader)

                # Print validation loss
                print(f"\t=> Validation Generator Loss: {val_loss_gen:.6f}")
                print(f"\t=> Validation Discriminator Loss: {val_loss_discr:.6f}")

            # Validation Part
            Validation(val_dataloader)
            
            # Save the model checkpoints at the end of the epoch in the save_path 
            if save_path is not None:
                save_obj = {}
                save_obj['text_emb_model']=self.text_emb_model
                save_obj['p_emb_dim']=self.p_emb_dim
                save_obj['embedding_size']=self.embedding_size

                save_obj["epoch"] = starting_epoch+epoch+1
                save_obj["total_steps"] = total_steps+1
                save_obj["tr_loss_gen"] = epoch_loss_gen
                save_obj["tr_loss_discr"] = epoch_loss_discr
                save_obj["gen_model_state_dict"] = self.generator.state_dict()
                save_obj["gen_optimizer_state_dict"] = self.optimG.state_dict()
                save_obj["discr_model_state_dict"] = self.discriminator.state_dict()
                save_obj["discr_optimizer_state_dict"] = self.optimD.state_dict()
                    
                torch.save(save_obj, os.path.join(save_path, f"{self.__class__.__name__}_epoch-{starting_epoch+epoch+1}.pt"))
                
                
    # Implement prediction  
    def predict(self, captions):
        """
        Generates a fake images based on the given captions using the generator model.

        Parameters:
            captions (str): The captions to generate the fake images from.

        Returns:
            torch.Tensor: The generated fake images. The shape of the tensor is (N, 3, H, W), 
            where H and W are the height and width of the image. The values are in the range [0, 1].
            N is the batch size 
        """
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            
            captions = captions.to(self.device)
            captions_embeddings = self.text_emb_model.encode_text(captions).to(self.device).float()   
            z = torch.randn(captions_embeddings.size(0), self.noise_dim, 1, 1).to(self.device).float()  
                
            fake_images = self.generator(captions_embeddings, z)     
            
        return fake_images
                
                
        
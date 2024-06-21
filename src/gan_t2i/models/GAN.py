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

class WGAN(nn.Module):
    
    def __init__(self, text_emb_model , embedding_size , p_emb_dim , device=("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(WGAN, self).__init__()
        
        
        self.device = device
        self.text_emb_model = text_emb_model
        
        # TODO : read parameters related to the GAN Net
        #with open('config.yaml', 'r') as f:
        #    config = yaml.load(f)
        self.noise_dim = 100
        self.batch_size = 32
        self.num_workers = 4
        self.lr = 0.05
        self.beta1 = 0.5
        self.DITER = 5               # DISCRIMINANT ITERATION
        self.l1_coef = 50
        self.l2_coef = 100
            
        # TODO: Extract embedding size from CLIP
        # coming from outside
        

        # TODO: Create generator and discriminator network 
        self.generator = torch.nn.DataParallel( Generator(embedding_size , p_emb_dim ) )
        self.discriminator = torch.nn.DataParallel( Discriminator(embedding_size , p_emb_dim ) )
        
        
        # TODO: load checkpoints coming from already executed net
        
        
    def fit(self, train_dataloader , val_dataloader , num_epochs , save_path ):
        
        # TODO : read info related to the net 
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
    


        print("Training on device: ", self.device)
        
        total_steps = 0                             # Counter of total number of iterations
        STEPS_PER_EPOCH = len(train_dataloader)
        
        # Set the logger if it is not provided
        #logger = logger or CSVLogger(os.path.join(os.getcwd(), "logs"), name=self._get_name())
        
        # Set up Discriminator and Generator Optimizer 
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        # Validate the checkpoint path or create it
        if save_path is not None and os.path.exists(save_path) and not os.path.isdir(save_path):
            raise ValueError(f"Checkpoint path {save_path} does not exist or isn't a directory.")
        elif save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)
        elif save_path is not None and os.path.exists(save_path) and os.path.isdir(save_path):
            for file in os.listdir(save_path):
                if file.endswith(".pt"):
                    raise ValueError(f"Checkpoint path {save_path} contains a checkpoint file, the checkpoint path must be empty ({file} remove it to continue).")
        
        # train WGAN
        one = torch.FloatTensor([1])
        mone = one * -1

        one = Variable(one).to(self.device)
        mone = Variable(mone).to(self.device)
        
        print("One : {} | Mone : {} ".format(one,mone))
        
        # TRAIN THE DISCRIMINATOR
        #for epoch in range(num_epochs):
        
        for epoch in range(self.num_epochs):
            print("epoch iter {} ".format(epoch))
            
            # Tell the model that we are training
            #self.generator.train()
            #self.discriminator.train()

            # Progress bar
            pbar = tqdm(total=STEPS_PER_EPOCH)
            running_loss_gen = 0.0
            running_loss_discr = 0.0
            
            for total_processed_batches, (images, captions, _) in enumerate(self.train_dataloader):
                images, captions = images.to(self.device), captions.to(self.device)
                
                #print("batch iter {} ".format(i))
                
                # Encode captions using the text_emb_model
                caption_embeddings = self.text_emb_model.encode_text(captions).to(self.device)
                #image_embeddings = self.text_emb_model.encode_image(images) #.to(self.device)
                
                #print("Dimensioni di caption_embeddings:", caption_embeddings.size())
                #print("Dimensioni di images:", images.size())
                #print("Dimensioni di images:", image_embeddings.size())

                # Extract Real Image and related Caption
                right_images = Variable(images.float()) #.to(self.device)
                right_embed = Variable(caption_embeddings.float())  #.to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                for j in range(self.DITER):                             # iterate on the descriminator 
                    self.optimD.zero_grad()

                    #print("Discriminator iteration {} ".format(j))
                    
                    # Generate salt
                    z = torch.randn(images.size(0), self.noise_dim, 1, 1).to(
                        self.device).float()
                    
                    # Generate Fake image based on salt
                    fake_images = self.generator(right_embed, z).detach()    # detach useful to not compute the gradient
                    wrong_images = Variable(fake_images.float()) #.to(self.device)
                    
                    #z = Variable(torch.randn(right_images.size(0), self.z_dim), volatile=True).to(self.device)
                    #z = z.view(z.size(0), self.z_dim, 1, 1)
                    #fake_images = Variable(self.generator(right_embed, z).data)

                    
                    #print("Free unsed space iter GENERATOR FAKE IMAGE {} ".format(i))
                    # Free up unused memory
                    torch.cuda.empty_cache()
                    
                    # Discriminate real image , real caption
                    d_real, _ = self.discriminator(right_images, right_embed)
                    d_real_loss = torch.mean(d_real)
                    d_real_loss.backward()
                    #d_real_loss.backward(mone)
                    
                    # Discriminate fake image , real caption
                    d_fake, _ = self.discriminator(wrong_images, right_embed)
                    d_fake_loss = torch.mean(d_fake)
                    d_fake_loss.backward()
                    #d_fake_loss.backward(one)
                    
                    # NOTE : Implement Gradient Penalty
                    
                    # compute of the loss function 
                    d_loss = d_real_loss - d_fake_loss
                    d_loss = - d_loss
                    self.optimD.step()
                    total_steps += 1
                    running_loss_discr += d_loss.item()
                    
                    # Weight clipping
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                        
                # ------------------------------------- #
                    
                #print("Summary Memory:",torch.cuda.memory_summary(device=None, abbreviated=False))
                #print("Free unsed space iter {} ".format(i))
                # Free up unused memory
                torch.cuda.empty_cache()
                
                
                # ---------------------
                #  Train Generator
                # ---------------------
                self.optimG.zero_grad()
                
                z = torch.randn(images.size(0), self.noise_dim, 1, 1).to(
                    self.device).float()
                
                # Generate Fake image
                fake_images = self.generator(right_embed , z)
                wrong_images = Variable(fake_images.float()) #.to(self.device)
                
                # Discriminate fake image , real caption
                d_fake, _ = self.discriminator(wrong_images, right_embed)
                
                g_loss = torch.mean(d_fake)
                #g_loss.backward(mone)
                g_loss.backward()
                #g_loss = - g_loss
                self.optimG.step()
                
                # Update counters
                total_steps += 1
                running_loss_gen += g_loss.item()
                
                # ------------------------ #
                
                # Free up unused memory
                torch.cuda.empty_cache()
                
                '''
                print(f"Epoch [{epoch+1}/{num_epochs}] | 
                      d_loss: {d_loss.item()} | g_loss: {g_loss.item()}")
                # Print the epoch summary on stdout
                print(f"\t=> Train Loss Generator: {g_loss:.6f}")
                print(f"\t=> Train Loss Discriminator: {d_loss:.6f}")
                print("")
                '''
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}] " + 
                                        f"Batch [{total_processed_batches+1}/{STEPS_PER_EPOCH}] \n " +
                                        f" => Loss Discriminator: {(running_loss_discr/(total_processed_batches+1)):.6f} \n" +
                                        f" => Loss Generator: {(running_loss_gen/(total_processed_batches+1)):.6f} \n " 
                                    )

            pbar.close()
            epoch_loss_gen = running_loss_gen / len(train_dataloader)
            epoch_loss_discr = running_loss_discr / len(train_dataloader)
            
            # Print the epoch summary after one epoch
            print(f"Epoch [{epoch+1}/{num_epochs}] Summary: \n")
            print(f"\t=> Train Generator Loss: {epoch_loss_gen:.6f} \n ")
            print(f"\t=> Train Discriminator Loss: {epoch_loss_discr:.6f} \n ")
            print("")
                
            # TODO: Validation Part
            
            # Save the model checkpoints at the end of the epoch in the save_path 
            # TODO:
            
            '''
            if save_path is not None:
                save_obj = {}
                save_obj["epoch"] = epoch+1
                save_obj["total_steps"] = total_steps+1
                save_obj["tr_loss_gen"] = epoch_loss_gen
                save_obj["tr_loss_discr"] = epoch_loss_discr
                save_obj["model_state_dict"] = self.state_dict()
                save_obj["optimizer_state_dict"] = optimizer.state_dict()
                if val_dataloader is not None:
                    save_obj["val_loss"] = total_val_loss
                    
                torch.save(save_obj, os.path.join(save_path, f"{self._get_name()}_epoch-{epoch+1}.pt"))
                
            # Print the epoch summary on stdout
            print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
            print(f"\t=> Train Loss: {epoch_loss:.6f}")
            if val_dataloader is not None:
                print(f"\t=> Val Loss: {total_val_loss:.6f}")
            print("")
            '''
                    
                    
            # How Visualize the an image for each batch
            #running_loss = 0.0
            #for total_processed_batches, (images, captions, class_number) in enumerate(train_dataloader, 0):
                
            # take the first image of the batch 
            #images = images[0]

            # Get the images and captions and move them to the device
            #images = images.to(self.device)
            #captions = captions.to(self.device).squeeze(1)
            
            '''
            print(type(images))
            print(type(captions))
            print(type(class_number))
            
            plt.figure(figsize=(6,6))
            plt.imshow(transforms.ToPILImage()(images))
            #plt.title(f"class: {class_number.numpy()}")    
            #print(f"image caption: {captions}")
            #plt.show()
            '''
            
            #self.optimG.zero_grad()
            #optimizer.zero_grad()   # Reset gradients
                
                
                
                
        
import os
import random

from torchinfo import summary
import torch.nn as nn
import torch

from tqdm import tqdm

from lightning.pytorch.loggers import CSVLogger


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


        # The generator network. 
        # Based on: https://github.com/aelnouby/Text-to-Image-Synthesis/blob/master/models/wgan_cls.py
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



    def forward(self, emb_cap, z):

        # Project the caption embedding 
        emb_proj = self.proj_emb_net(emb_cap)

        # Concatenate the projected caption embedding and the noise vector z
        latent_vector = torch.cat([emb_proj, z], 1)

        # Add a third and fourth dimension to the latent vector to match the dimension of the generator network
        # ConvTranspose2D since it needs a tensor with 4 dimensions (batch_size, channels, height, width) 
        latent_vector = latent_vector.unsqueeze(2).unsqueeze(3)

        # Generate the image from the latent vector
        return self.generator_net(latent_vector)





class Discriminator(nn.Module):
    
    def __init__(self, emb_dim, proj_emb_dim = 128, num_dis_features = 64):

        super(Discriminator, self).__init__()

        self.emb_dim = emb_dim
        self.proj_emb_dim = proj_emb_dim
        self.n_df = num_dis_features


        # The projection network of the caption embedding
        self.proj_emb_net = nn.Sequential(
            nn.Linear(
                in_features=self.emb_dim,
                out_features=self.proj_emb_dim
            ),
            nn.BatchNorm1d(num_features=self.proj_emb_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


        # The discriminator network. 
        # Based on: https://github.com/aelnouby/Text-to-Image-Synthesis/blob/master/models/wgan_cls.py
        self.discriminator_net_1 = nn.Sequential(

			# Input size => 3 x 64 x 64
			nn.Conv2d(3, self.n_df, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# Current size => (ndf) x 32 x 32
		
            nn.Conv2d(self.n_df, self.n_df * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.n_df * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# Current size => (ndf * 2) x 16 x 16
		
            nn.Conv2d(self.n_df * 2, self.n_df * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.n_df * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# Current size => (ndf * 4) x 8 x 8
		
            nn.Conv2d(self.n_df * 4, self.n_df * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.n_df * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# Current size => (ndf * 8) x 4 x 4
        )



        # The second discriminator network.
        self.discriminator_net_2 = nn.Sequential(

            # Input size => ((ndf * 8) + proj_emb_size) x 4 x 4
    	    nn.Conv2d(self.n_df * 8 + self.proj_emb_dim, 1, 4, 1, 0, bias=False),
            # Output size => 1 x 1 x 1
            nn.Sigmoid()
		)


    def forward(self, img, cap_emb):
        
        # Result of the first discriminator conv network
        x_int = self.discriminator_net_1(img)

        # Caption embedding projection        
        proj_emb = self.proj_emb_net(cap_emb)
        
        # Resizing the projected embedding repating it 
        # 4 times to the dimensione (n_batch x proj_emb_dim x 4 x 4) 
        rep_emb = proj_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)

        # Concatenate the resized and repated projected embedding 
        # with the result of the first discriminator conv network
        x_concat = torch.cat([x_int, rep_emb], 1)

        # Forward the concatenated tensor to the second discriminator conv network
        x = self.discriminator_net_2(x_concat).squeeze(1).squeeze(1).squeeze(1)

        return x




class GAN_INT_CLS(nn.Module):
    
    def __init__(self, 
                 emb_network, 
                 emb_dim, 
                 proj_emb_dim = 128, 
                 noise_dim = 100, 
                 num_gen_features = 64, 
                 device = "cuda" if torch.cuda.is_available() else "cpu"
                ):
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

        self.device = device
        self.emb_dim = emb_dim
        self.proj_emb_dim = proj_emb_dim
        self.noise_dim = noise_dim
        self.num_gen_features = num_gen_features


        # The embedding network
        self.emb_net = emb_network.to(device)

        # The generator network
        self.generator = Generator(
            emb_dim = emb_dim,
            proj_emb_dim = proj_emb_dim,
            noise_dim = noise_dim,
            num_gen_features = num_gen_features
        ).to(device)

        # The discriminator network
        self.discriminator = Discriminator(
            emb_dim = emb_dim,
            proj_emb_dim = proj_emb_dim,
            num_dis_features = num_gen_features
        ).to(device)
        

    def generate_images(self, captions):
        if len(captions.size()) == 1:
            captions = captions.unsqueeze(0)
        
        with torch.no_grad():
            self.generator.eval()
            emb_cap = self.emb_net.encode_text(captions.to(self.device)).to(self.device).float() 
            z = torch.distributions.Normal(0.0, 0.1).sample((captions.size(0), self.noise_dim)).to(self.device)
            fake_images = self.generator(emb_cap, z)
            
            return fake_images


    def fit(self, 
            train_dataloader, val_dataloader = None, 
            num_epochs = 600,
            gen_optim = None, 
            disc_optim = None, 
            device = "cuda" if torch.cuda.is_available() else "cpu", 
            save_path = None,
            logger = None,
            ):
        """ 
            # Fit the model
            Train the model.
            
            ## Args:
            - train_dataloader (torch.utils.data.DataLoader): train dataloader.
            - val_dataloader (torch.utils.data.DataLoader, optional): validation dataloader. Defaults to None.
            - num_epochs (int, optional): number of epochs. Defaults to 10.
            - gen_optim (torch.optim, optional): optimizer. Defaults to torch.optim.Adam with (lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4).
            - disc_optim (torch.optim, optional): optimizer. Defaults to torch.optim.Adam with (lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4).
            - device (str, optional): device. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
            - save_path (str, optional): path where to save the checkpoints. Defaults to None.
            - logger (lightning.pytorch.loggers.Logger, optional): logger. Defaults to CSVLogger with log path ${curr_working_dir}/logs}.
        """

        self.generator.to(device)
        self.discriminator.to(device)
        self.emb_net.to(device)

        total_steps = 0        # Counter of total number of iterations
        STEPS_PER_EPOCH = len(train_dataloader)

        logger = logger or CSVLogger(os.path.join(os.getcwd(), "logs"), name=self._get_name())

        if gen_optim is None:
            gen_optim = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        if disc_optim is None:
            disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)


        criterion = nn.BCELoss()


        logger.log_hyperparams({"epochs": num_epochs, 
                                "gen optimizer": type(gen_optim), "gen optimizer_kwargs": gen_optim.defaults, 
                                "disc optimizer": type(disc_optim), "disc optimizer_kwargs": disc_optim.defaults, 
                                "loss": criterion.__class__.__name__})


        def sample_wrong_embeddings(dataloader, classes):
            # Sample wrong captions from the dataset
                wrong_cap = []
                for i in range(emb_cap.size(0)):
                    _, w_cap, w_class = dataloader.dataset[random.randint(0, len(dataloader.dataset)-1)]
                    while w_class == classes[i]:
                        _, w_cap, w_class = dataloader.dataset[random.randint(0, len(dataloader.dataset)-1)]
                    wrong_cap.append(w_cap)
                
                wrong_cap = torch.stack(wrong_cap).to(device)
                with torch.no_grad():
                    wrong_emb = self.emb_net.encode_text(wrong_cap).to(device).float()

                return wrong_emb

        def process_batch(dataloader, images, emb_cap, classes):

            # Sample wrong captions from the dataset
            wrong_emb = sample_wrong_embeddings(dataloader, classes)

            # Labels for BCE loss (ones for real, zeros for fake) 
            labels_ones = torch.ones(images.size(0)).to(device)
            labels_zeros = torch.zeros(images.size(0)).to(device)

            # ***********************
            # GENERATOR
            # ***********************
            z = torch.distributions.Normal(0.0, 0.1).sample((images.size(0), self.noise_dim)).to(device)
            fake_images = self.generator(emb_cap, z)
            
            # ***********************
            # DISCRIMINATOR
            # ***********************
            # Right image, Right embedding
            real_out = self.discriminator(images, emb_cap)
            real_loss = criterion(real_out, labels_ones)
            real_acc = (real_out >= 0.5).float().sum().item() / real_out.size(0)

            # Fake image, Right embedding
            fake_out = self.discriminator(fake_images, emb_cap)
            fake_loss = criterion(fake_out, labels_zeros)
            fake_acc = (fake_out <= 0.5).float().sum().item() / fake_out.size(0)

            # Right image, Wrong embedding
            wrong_out = self.discriminator(images, wrong_emb)
            wrong_loss = criterion(wrong_out, labels_zeros)
            wrong_acc = (wrong_out <= 0.5).float().sum().item() / wrong_out.size(0)

            return real_loss, fake_loss, wrong_loss, real_acc, fake_acc, wrong_acc
        

        for epoch in range(num_epochs):
            
            # Tell the model that we are training
            self.generator.train()
            self.discriminator.train()

            # Progress bar
            pbar = tqdm(total=STEPS_PER_EPOCH)

            running_loss = 0.0
            running_real_acc, running_fake_acc, running_wrong_acc = 0.0, 0.0, 0.0
            for total_processed_batches, (images, captions, classes) in enumerate(train_dataloader):
                
                images = images.to(device)          # Images to device
                captions = captions.to(device)      # Captions to device

                with torch.no_grad():
                    emb_cap = self.emb_net.encode_text(captions).to(device).float()     # Encode captions embeddings
                
                gen_optim.zero_grad()
                disc_optim.zero_grad()

                real_loss, fake_loss, wrong_loss, real_acc, fake_acc, wrong_acc = process_batch(train_dataloader, images, emb_cap, classes)
                loss = real_loss + fake_loss + wrong_loss
                
                running_loss += loss.item()
                running_real_acc +=  real_acc
                running_fake_acc += fake_acc
                running_wrong_acc += wrong_acc

                loss.backward()
                disc_optim.step()
                gen_optim.step()

                logger.log_metrics({
                    "iteration": total_steps+1, 
                    "real_loss": real_loss.item(), 
                    "fake_loss": fake_loss.item(), 
                    "wrong_loss": wrong_loss.item(),
                    "real_acc": real_acc,
                    "fake_acc": fake_acc,
                    "wrong_acc": wrong_acc,
                    "loss": loss.item()
                }, step=total_steps)

                total_steps += 1
                pbar.update(1)

                pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}] " + 
                                     f"Batch [{total_processed_batches+1}/{STEPS_PER_EPOCH}]: " +
                                     f"Loss: {(running_loss/(total_processed_batches+1)):.6f} |" + 
                                     f"Real Acc.: {(running_real_acc/(total_processed_batches+1)):.6f} |" + 
                                     f"Fake Acc.: {(running_fake_acc/(total_processed_batches+1)):.6f} |" + 
                                     f"Wrong Acc.: {(running_wrong_acc/(total_processed_batches+1)):.6f}")
    
            pbar.close()


            # Compute the metrics and log them
            epoch_loss = running_loss / len(train_dataloader)
            running_real_acc = running_real_acc / len(train_dataloader)
            running_fake_acc = running_fake_acc / len(train_dataloader)
            running_wrong_acc = running_wrong_acc / len(train_dataloader)

            metrics = {"epoch": epoch+1, "train_loss": epoch_loss, "train_real_acc": running_real_acc, "train_fake_acc": running_fake_acc, "train_wrong_acc": running_wrong_acc}
            
            # If the val_dataloader is not None, evaluate the model on the validation set
            if val_dataloader is not None:
                # Tell the model that we are evaluating
                self.generator.eval()
                self.discriminator.eval()
            
                with torch.no_grad():        
                    total_val_loss = 0.0
                    val_running_real_acc, val_running_fake_acc, val_running_wrong_acc = 0.0, 0.0, 0.0
                    v_bar = tqdm(val_dataloader)
                    for idx, (images, captions, classes) in enumerate(val_dataloader):                        
                        
                        images = images.to(device)
                        captions = captions.to(device).squeeze(1)
                        
                        
                        emb_cap = self.emb_net.encode_text(captions).to(device).float()     # Encode captions embeddings
                        
                        real_loss, fake_loss, wrong_loss, real_acc, fake_acc, wrong_acc = process_batch(val_dataloader, images, emb_cap, classes)
                        val_loss = real_loss + fake_loss + wrong_loss

                        total_val_loss += val_loss.item()
                        val_running_real_acc +=  real_acc
                        val_running_fake_acc += fake_acc
                        val_running_wrong_acc += wrong_acc

                        v_bar.update(1)
                        v_bar.set_description(f"Validation => Val Loss: {total_val_loss / (idx+1):.6f}" + 
                                              f" | Real Acc.: {val_running_real_acc / (idx+1):.6f}" + 
                                              f" | Fake Acc.: {val_running_fake_acc / (idx+1):.6f}" + 
                                              f" | Wrong Acc.: {val_running_wrong_acc / (idx+1):.6f}")
                    
                    v_bar.close()

                    # Compute the validation loss and add it to the metrics
                    total_val_loss = total_val_loss / len(val_dataloader)
                    val_running_real_acc = val_running_real_acc / len(val_dataloader)
                    val_running_fake_acc = val_running_fake_acc / len(val_dataloader)
                    val_running_wrong_acc = val_running_wrong_acc / len(val_dataloader)

                    metrics["val_loss"] = total_val_loss
                    metrics["val_real_acc"] = val_running_real_acc 
                    metrics["val_fake_acc"] = val_running_fake_acc 
                    metrics["val_wrong_acc"] = val_running_wrong_acc 

            logger.log_metrics(metrics)
            logger.save()
            
            # Save the checkpoint
            if save_path is not None:
                save_obj = {}
                save_obj["epoch"] = epoch+1
                save_obj["total_steps"] = total_steps+1
                save_obj["tr_loss"] = epoch_loss
                save_obj["model_state_dict"] = self.state_dict()
                save_obj["gen_optimizer_state_dict"] = gen_optim.state_dict()
                save_obj["disc_optimizer_state_dict"] = disc_optim.state_dict()
                if val_dataloader is not None:
                    save_obj["val_loss"] = total_val_loss
                    
                torch.save(save_obj, os.path.join(save_path, f"{self._get_name()}_epoch-{epoch+1}.pt"))
                
            # Print the epoch summary on stdout
            print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
            print(f"  - Train")
            print(f"    => Total Loss: {epoch_loss:.6f}")
            print(f"    => Real Acc: {running_real_acc:.6f}")
            print(f"    => Fake Acc: {running_fake_acc:.6f}")
            print(f"    => Wrong Acc: {running_wrong_acc:.6f}")

            if val_dataloader is not None:
                print(f"  - Validation")
                print(f"    => Val Loss: {total_val_loss:.6f}")
                print(f"    => Real Acc: {val_running_real_acc:.6f}")
                print(f"    => Fake Acc: {val_running_fake_acc:.6f}")
                print(f"    => Wrong Acc: {val_running_wrong_acc:.6f}")
            print("")



    def summary(self, batch_size = 1):
        """ 
        # Summary
        Prints the summary of the model 
        """

        # Generator summary
        summary(self.generator, [(batch_size, self.emb_dim), (batch_size, self.noise_dim)])

        # Discriminator summary
        summary(self.discriminator, [(batch_size, 3, 64, 64), (batch_size, self.emb_dim)])
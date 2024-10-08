import torch
import os
from torch.autograd import Variable
from tqdm import tqdm

from gan_t2i.models.Generator import Generator
from gan_t2i.models.Discriminator import Discriminator

from ..utils import logger


class WGAN(object):

    def __init__(
        self,
        text_emb_model,
        embedding_size,
        p_emb_dim=128,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        WGAN

        Args:
            text_emb_model: model used for text embedding
            embedding_size (int): output dimension of the last layer of the text_emb_model
            p_emb_dim (int): dimension which we want to project the text embedding (default: 128)
            device (str, optional): device to use. Defaults to "cuda" if CUDA is available, otherwise "cpu".
        """

        super(WGAN, self).__init__()

        # Initialize parameters
        self.device = device
        self.text_emb_model = text_emb_model
        self.embedding_size = embedding_size
        self.p_emb_dim = p_emb_dim

        # Other model parameters
        self.noise_dim = 100  # Dimension of the noise z
        self.DITER = 5        # Discriminator interation (for same batch)

        # Optimizer parameters
        self.lr = 0.05
        self.beta1 = 0.5

        # Create generator and discriminator network
        self.generator = torch.nn.DataParallel(Generator(embedding_size, p_emb_dim))
        self.discriminator = torch.nn.DataParallel(Discriminator(embedding_size, p_emb_dim))

        # Discriminator optimizer
        self.optimD = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )

        # Generator optimizer
        self.optimG = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )

    @staticmethod
    def load(
        model_pt_filepath, device=("cuda" if torch.cuda.is_available() else "cpu")
    ):
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
            raise FileNotFoundError(
                f"Checkpoint path {model_pt_filepath} does not exist"
            )

        # Load the checkpoint
        checkpoint = torch.load(model_pt_filepath, map_location=device)

        # Load model parameters
        text_emb_model = checkpoint["text_emb_model"]
        embedding_size = checkpoint["embedding_size"]
        p_emb_dim = checkpoint["p_emb_dim"]

        # Create the model
        loaded_model = WGAN(text_emb_model, embedding_size, p_emb_dim)

        # Loading generator state dict in the model
        loaded_model.generator.load_state_dict(checkpoint["gen_model_state_dict"])
        loaded_model.optimG.load_state_dict(checkpoint["gen_optimizer_state_dict"])

        # Loading discriminator state dict in the model
        loaded_model.discriminator.load_state_dict(checkpoint["discr_model_state_dict"])
        loaded_model.optimD.load_state_dict(checkpoint["discr_optimizer_state_dict"])

        logger.info(f"Checkpoint loaded for epoch {checkpoint["epoch"]}.")

        return loaded_model

    def fit(self, train_dataloader, val_dataloader, num_epochs, save_path, starting_epoch=0):

        def __get_starting_epoch(save_path: str, starting_epoch: int) -> int:
            if save_path is None:
                return 0

            # if the save_path does not exists create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                return 0

            # if the save_path is not a directory
            if not os.path.isdir(save_path):
                raise ValueError(f"Checkpoint path {save_path} does not exist or isn't a directory.")

            # Check if the save_path has a .pt (checkpoint) file
            has_pt: bool = False
            for file in os.listdir(save_path):
                if file.endswith(".pt"):
                    has_pt = True
                    break

            # if no .pt files are present in save_path then start the training from the epoch 0
            if not has_pt:
                return 0

            if starting_epoch < 0:
                raise ValueError(f"Starting epoch {starting_epoch} is not valid.")

            logger.warning(f"Checkpoints are present in '{save_path}', training will start from epoch {starting_epoch}")
            return starting_epoch

        def __run_validation(val_dataloader):
                if not val_dataloader:
                    return

                # Set the generator and discriminator to evaluation mode
                self.generator.eval()
                self.discriminator.eval()

                # Initialize variables for generator and discriminator validation losses
                val_loss_gen = 0.0
                val_loss_discr = 0.0

                # Disabling gradients during validation
                with torch.no_grad():

                    # Progress bar
                    pbar = tqdm(total=len(val_dataloader))

                    for val_tot_processed_batches, (val_images, val_captions, _) in enumerate(val_dataloader):

                        val_images, val_captions = val_images.to(self.device), val_captions.to(self.device)

                        # Encode captions using the text embedding model
                        val_captions_embeddings = self.text_emb_model.encode_text(val_captions).to(self.device)

                        # Prepare validation images and captions embeddings
                        right_images = val_images.float().to(self.device)
                        right_embeds = val_captions_embeddings.float().to(self.device)

                        # Generate fake images
                        z = torch.randn(val_images.size(0), self.noise_dim, 1, 1).to(self.device).float()
                        fake_images = self.generator(right_embeds, z)

                        # Discriminate with: real image, real caption
                        d_real, _ = self.discriminator(right_images, right_embeds)
                        d_real_loss = torch.mean(d_real)

                        # Discriminator with: fake image, real caption
                        d_fake, _ = self.discriminator(fake_images, right_embeds)
                        d_fake_loss = torch.mean(d_fake)

                        # Compute losses for the generator and discriminator
                        d_loss = d_real_loss - d_fake_loss
                        g_loss = d_fake_loss

                        # Add the losses
                        val_loss_discr += d_loss.item()
                        val_loss_gen += g_loss.item()

                        # Update progress bar
                        pbar.update(1)
                        pbar.set_description(
                            f"Validation [{val_tot_processed_batches+1}/{len(val_dataloader)}] - " +
                            f"Generator loss: {(val_loss_gen/(val_tot_processed_batches+1)):.6f} | " +
                            f"Discriminator loss: {(val_loss_discr/(val_tot_processed_batches+1)):.6f}"
                        )

                    pbar.close()

                # Compute the validation loss
                val_loss_gen /= len(val_dataloader)
                val_loss_discr /= len(val_dataloader)

                # Print validation loss
                logger.info(f"\t=> Validation Generator Loss: {val_loss_gen:.6f}")
                logger.info(f"\t=> Validation Discriminator Loss: {val_loss_discr:.6f}")

        total_steps = 0                              # Counter of total number of steps/iterations
        steps_per_epoch = len(train_dataloader) # Number of steps/iterations per epoch
        starting_epoch = __get_starting_epoch(save_path, starting_epoch)   # Get the starting epoch
        num_epochs = starting_epoch + num_epochs

        # WGAN training parameters
        one = torch.FloatTensor([1])        # Tesor of values one (1)
        mone = one * -1             # Tensor of values minus one (-1)
        one = Variable(one).to(self.device)
        mone = Variable(mone).to(self.device)

        logger.info(f"Training on device: {self.device}")

        for epoch in range(starting_epoch, num_epochs):

            # Progress bar
            pbar = tqdm(total=steps_per_epoch)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            # Running generator and discriminator losses
            running_loss_gen = 0.0
            running_loss_discr = 0.0

            for tot_prcessed_batches, (images, captions, _) in enumerate(train_dataloader):

                images, captions = images.to(self.device), captions.to(self.device)

                # Encode captions using the text_emb_model
                captions_embeddings = self.text_emb_model.encode_text(captions).to(self.device)

                # Make images and capiotns embeddings float
                right_images = Variable(images.float()).to(self.device)
                right_embeds = Variable(captions_embeddings.float()).to(self.device)

                # Set the parameters discriminator to trainable
                for p in self.discriminator.parameters():
                    p.requires_grad = True

                # Discriminator training
                self.discriminator.train()

                for j in range(self.DITER):

                    # Reset the optimizer gradient
                    self.optimD.zero_grad()

                    # Free up unused memory
                    torch.cuda.empty_cache()

                    # Generate random noise/salt
                    z = torch.randn(images.size(0), self.noise_dim, 1, 1).to(self.device).float()

                    # Generate Fake image based on noise/salt
                    # Detach:  does not compute the gradient in the generator network since we are not training the generator now
                    fake_images = self.generator(right_embeds, z).detach()
                    fake_images = Variable(fake_images.float())

                    # Discriminator with: real image, real caption
                    d_real, _ = self.discriminator(right_images, right_embeds)
                    d_real_loss = torch.mean(d_real).unsqueeze(0)
                    d_real_loss.backward(mone)

                    # Discriminate with: fake image, real caption
                    d_fake, _ = self.discriminator(fake_images, right_embeds)
                    d_fake_loss = torch.mean(d_fake).unsqueeze(0)
                    d_fake_loss.backward(one)

                    # Compute the difference between real and fake losses
                    d_diff_loss = d_real_loss - d_fake_loss
                    running_loss_discr += d_diff_loss.item()

                    # Optimizer step
                    self.optimD.step()

                    # Weight clipping
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                # Free up unused memory
                torch.cuda.empty_cache()

                # Train the generator
                self.generator.train()

                # Reset generator optimizer gradient
                self.optimG.zero_grad()

                # Generate random noise/salt
                z = torch.randn(images.size(0), self.noise_dim, 1, 1).to(self.device).float()

                # Generate Fake image
                fake_images = self.generator(right_embeds, z)
                fake_images = Variable(fake_images.float())

                # Discriminator with: fake image, real caption
                d_fake, _ = self.discriminator(fake_images, right_embeds)

                # Compute the generator loss
                g_loss = torch.mean(d_fake).unsqueeze(0)
                g_loss.backward(mone)
                g_loss = -g_loss

                # Optimizer step
                self.optimG.step()

                # Update counters
                total_steps += 1
                running_loss_gen += g_loss.item()

                # Free up unused memory
                torch.cuda.empty_cache()

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    + f"Batch [{tot_prcessed_batches+1}/{steps_per_epoch}] "
                    + f"Generator loss: {(running_loss_gen/(tot_prcessed_batches+1)):.6f} | "
                    + f"Discriminator loss: {(running_loss_discr/(tot_prcessed_batches+1)):.6f}"
                )

            pbar.close()

            # Compute the epoch loss
            epoch_loss_gen = running_loss_gen / len(train_dataloader)
            epoch_loss_discr = running_loss_discr / len(train_dataloader)

            # Print the epoch summary after one epoch
            logger.info(f"Epoch [{starting_epoch+epoch+1}/{num_epochs}] Summary:")
            logger.info(f"\t=> Train Generator Loss: {epoch_loss_gen:.6f}")
            logger.info(f"\t=> Train Discriminator Loss: {epoch_loss_discr:.6f}")

            # Validation Part
            __run_validation(val_dataloader)

            # Save the model checkpoints at the end of the epoch in the save_path
            if save_path is not None:
                save_obj = {}
                save_obj["text_emb_model"] = self.text_emb_model
                save_obj["p_emb_dim"] = self.p_emb_dim
                save_obj["embedding_size"] = self.embedding_size

                save_obj["epoch"] = starting_epoch + epoch + 1
                save_obj["total_steps"] = total_steps + 1
                save_obj["tr_loss_gen"] = epoch_loss_gen
                save_obj["tr_loss_discr"] = epoch_loss_discr
                save_obj["gen_model_state_dict"] = self.generator.state_dict()
                save_obj["gen_optimizer_state_dict"] = self.optimG.state_dict()
                save_obj["discr_model_state_dict"] = self.discriminator.state_dict()
                save_obj["discr_optimizer_state_dict"] = self.optimD.state_dict()

                new_pt_file = f"{self.__class__.__name__}_epoch-{epoch+1}.pt"
                torch.save(save_obj, os.path.join(save_path, new_pt_file))


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
            captions_embeddings = (
                self.text_emb_model.encode_text(captions).to(self.device).float()
            )
            z = (
                torch.randn(captions_embeddings.size(0), self.noise_dim, 1, 1)
                .to(self.device)
                .float()
            )

            fake_images = self.generator(captions_embeddings, z)

        return fake_images

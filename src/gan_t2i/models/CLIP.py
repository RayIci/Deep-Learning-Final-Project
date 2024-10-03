import clip

import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger

from tqdm import tqdm

import os


def ContrastiveLoss(image_features, text_features, temperature=0.01, device=("cuda" if torch.cuda.is_available() else "cpu")):
    """ 
    # Contrastive Loss Function
    The contrastive loss is calculated as the sum of two cross-entropy terms, 
    one for images and one for texts. Each term attempts to correctly align the 
    corresponding image-text pairs. The formula can be written as:

    Loss = (1/2) * (Loss_img + Loss_txt)
    
    where Loss_img and Loss_txt are the cross-entropy losses for the image and 
    text embeddings, respectively.
    
    ## Args:
    - image_features (torch.Tensor): image features
    - text_features (torch.Tensor): text features
    - temperature (float, optional): controls the smoothness. Defaults to 0.01.
    """
    
    logits_img = image_features @ text_features.t() / temperature
    logits_txt = text_features @ image_features.t() / temperature
    
    labels = torch.arange(len(image_features)).to(device)
    
    loss_img = F.cross_entropy(logits_img, labels)
    loss_txt = F.cross_entropy(logits_txt, labels)
    
    return (loss_img + loss_txt) / 2


class CLIPModel(torch.nn.Module):
    
    def get_output_dimensions(self):
        """Function to get the output dimensions of the CLIP model for both images and texts."""
        dummy_image = torch.zeros(1, 3, 224, 224).to(self.device)  # Dummy image tensor
        dummy_text = clip.tokenize(["dummy text"]).to(self.device)  # Dummy text tensor
        
        with torch.no_grad():
            image_features = self.encode_image(dummy_image)
            text_features = self.encode_text(dummy_text)
        
        return image_features.shape, text_features.shape
    
    @staticmethod
    def load(model_pt_filepath, device=("cuda" if torch.cuda.is_available() else "cpu")):

        """ 
        # Load CLIP
        Load a CLIP model from a pt file containing the model state_dict.
        
        ## Args:
        - model_pt_filepath (str): path to the pt file containing the model state_dict.
        - device (str, optional): device to use. Defaults to ("cuda" if torch.cuda.is_available() else "cpu").
        
        ## Returns:
        - CLIPModel: the loaded model
        """

        checkpoint = torch.load(model_pt_filepath,map_location=device)
        loaded_model = CLIPModel(device=device)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        return loaded_model
    
    def __init__(self, model_name="ViT-B/32", device=("cuda" if torch.cuda.is_available() else "cpu")):
        """ 
        # CLIP Model
        The model is based on the CLIP library.
        
        ## Args:
        - model_name (str, optional): model name or the path to a model checkpoint containing the state_dict. Defaults to "ViT-B/32".
        - device (str, optional): device to use. Defaults to ("cuda" if torch.cuda.is_available() else "cpu").
        """
        
        super().__init__()
        
        self.model, _ = clip.load(model_name)
        self.device = device

        self.model.to(self.device)
        self.to(self.device)                
        print(f"Model loaded on device: {device}")
        
    def forward(self, images, texts):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        return image_features, text_features
        
    def encode_image(self, images):
        return self.model.encode_image(images)
    
    def encode_text(self, texts):
        return self.model.encode_text(texts)
    
    def fit(self, 
        train_dataloader,
        num_epochs = 10,
        optimizer = None,
        val_dataloader = None,
        loss_function = ContrastiveLoss,
        loss_kwargs = {},
        save_path = None,
        logger = None,
        starting_epoch =0 
    ):
        """ 
        # Fit the model
        Train the model.
        
        ## Args:
        - train_dataloader (torch.utils.data.DataLoader): train dataloader.
        - num_epochs (int, optional): number of epochs. Defaults to 10.
        - optimizer (torch.optim, optional): optimizer. Defaults to torch.optim.AdamW with lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2.
        - val_dataloader (torch.utils.data.DataLoader, optional): validation dataloader. Defaults to None.
        - loss (Callable, optional): loss function. Defaults to ContrastiveLoss.
        - loss_kwargs (dict, optional): loss function keyword arguments. Defaults to {}.
        - save_path (str, optional): path where to save the checkpoints. Defaults to None.
        - logger (lightning.pytorch.loggers.Logger, optional): logger. Defaults to CSVLogger with log path ${curr_working_dir}/logs}.
        """
        
        print("Training on device: ", self.device)
        
        total_steps = 0        # Counter of total number of iterations
        STEPS_PER_EPOCH = len(train_dataloader)
        
        # Set the logger if it is not provided
        logger = logger or CSVLogger(os.path.join(os.getcwd(), "logs"), name=self._get_name())
        
        # Set the optimizer if it is not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, betas=(0.9,  0.98), eps=1e-6, weight_decay=0.2)
        
        # Validate the checkpoint path or create it
        if save_path is not None and os.path.exists(save_path) and not os.path.isdir(save_path):
            raise ValueError(f"Checkpoint path {save_path} does not exist or isn't a directory.")
        elif save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)
        elif save_path is not None and os.path.exists(save_path) and os.path.isdir(save_path):
            for file in os.listdir(save_path):
                if file.endswith(".pt"):
                    print("The checkpoint_CLIP folder already contains .pt's file with related epochs , we start from {} to train ".format(starting_epoch))
                    #raise ValueError(f"Checkpoint path {save_path} contains a checkpoint file, the checkpoint path must be empty ({file} remove it to continue).")
        
        logger.log_hyperparams({"epochs": num_epochs, "optimizer": type(optimizer), "optimizer_kwargs": optimizer.defaults, "loss": loss_function.__name__, "loss_kwargs": loss_kwargs})
        
        
        def process_batch(images, captions):
            # Get the image and text features
            image_features = self.model.encode_image(images).to(self.device)
            text_features = self.model.encode_text(captions).to(self.device)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute the loss
            return loss_function(image_features, text_features, **loss_kwargs)
        
        
        
        for epoch in range(num_epochs):
            
            # Tell the model that we are training
            self.model.train()

            # Progress bar
            pbar = tqdm(total=STEPS_PER_EPOCH)
                
            running_loss = 0.0
            for total_processed_batches, (images, captions, _) in enumerate(train_dataloader, 0):
                
                # Get the images and captions and move them to the device
                images = images.to(self.device)
                captions = captions.to(self.device).squeeze(1)

                optimizer.zero_grad()   # Reset gradients

                loss = process_batch(images, captions)  # compute the loss for the batch
                
                logger.log_metrics({"iteration": total_steps+1, "loss": loss}, step=total_steps)
                
                # Backpropagate and update weights
                loss.backward()
                if self.device == "cpu":
                    optimizer.step()
                else: 
                    self.model.float()          # Need to convert models to float for training
                    optimizer.step()            # This is due to an issue: https://github.com/openai/CLIP/issues/83
                    clip.model.convert_weights(self.model)

                    
                # Update counters
                total_steps += 1
                running_loss += loss.item()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Epoch [{starting_epoch+ epoch+1}/{starting_epoch+num_epochs}] " + 
                                     f"Batch [{total_processed_batches+1}/{STEPS_PER_EPOCH}]: " +
                                     f"Loss: {(running_loss/(total_processed_batches+1)):.6f}")

            pbar.close()
            
            # Compute the metrics and log them
            epoch_loss = running_loss / len(train_dataloader)
            metrics = {"epoch": starting_epoch+epoch+1, "train_loss": epoch_loss,}
            
            # If the val_dataloader is not None, evaluate the model on the validation set
            if val_dataloader is not None:
                # Tell the model that we are evaluating
                self.model.eval()
            
                with torch.no_grad():        
                    total_val_loss = 0.0
                    for images, captions, _ in tqdm(val_dataloader, desc="\t=> Validation: "):                        
                        images = images.to(self.device)
                        captions = captions.to(self.device).squeeze(1)
                        total_val_loss += process_batch(images, captions).item()

                    # Compute the validation loss and add it to the metrics
                    total_val_loss = total_val_loss / len(val_dataloader)
                    metrics["val_loss"] = total_val_loss

            logger.log_metrics(metrics)
            logger.save()
            
            # Save the checkpoint
            if save_path is not None:
                save_obj = {}
                save_obj["epoch"] = starting_epoch+epoch+1
                save_obj["total_steps"] = total_steps+1
                save_obj["tr_loss"] = epoch_loss
                save_obj["model_state_dict"] = self.state_dict()
                save_obj["optimizer_state_dict"] = optimizer.state_dict()
                if val_dataloader is not None:
                    save_obj["val_loss"] = total_val_loss
                    
                torch.save(save_obj, os.path.join(save_path, f"{self._get_name()}_epoch-{starting_epoch+epoch+1}.pt"))
                
            # Print the epoch summary on stdout
            print(f"Epoch [{starting_epoch + epoch+1}/{num_epochs}] Summary:")
            print(f"\t=> Train Loss: {epoch_loss:.6f}")
            if val_dataloader is not None:
                print(f"\t=> Val Loss: {total_val_loss:.6f}")
            print("")
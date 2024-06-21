import os

import clip

from gan_t2i.models.GAN_INT_CLS import GAN_INT_CLS
from gan_t2i.datasets.DatasetFactory import DatasetFactory
from gan_t2i.models.CLIP import CLIPModel
from gan_t2i.utils.model_loading import download_CLIP_model, CLIP_DATASETS

import torch 
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from PIL import Image


# Images transformations
transform_img = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),    
    transforms.Normalize([0.4355, 0.3777, 0.2879], [0.2571, 0.2028, 0.2101])
])

# Text transformation (caption tokenizer)
def tokenize_text(text):
    try:
        return clip.tokenize([text])[0]
    except:
        return clip.tokenize([text.split(".")[0]])[0] 


# Load data
data_path = os.path.join(os.getcwd(), "data") 
dataset = DatasetFactory.Flowers(data_path, transform_img=transform_img, transform_caption=tokenize_text)

# Split data into train, validation and test
t_size = int(0.85 * len(dataset))
v_size = int(0.1 * len(dataset))
t_size = len(dataset) - v_size - t_size

train_sampler = SubsetRandomSampler(list(range(t_size)))
val_sampler = SubsetRandomSampler(list(range(t_size, t_size + v_size)))
test_sampler = SubsetRandomSampler(list(range(t_size + v_size, len(dataset))))

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, pin_memory=True)
val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler, pin_memory=True) 

# Get the caption embedding network (our finetuned version of CLIP)
checkpoint_path = download_CLIP_model(CLIP_DATASETS.FLOWERS)
clip_emb_model = CLIPModel.load(checkpoint_path)

# Create the GAN-INT-CLS model
gan = GAN_INT_CLS(
    emb_network=clip_emb_model,
    emb_dim=512
)
gan.summary()

# gan.fit(train_loader, val_loader)
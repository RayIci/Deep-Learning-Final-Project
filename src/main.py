import os

import clip

from gan_t2i.models.GAN_INT_CLS import GAN_INT_CLS
from gan_t2i.datasets.DatasetFactory import DatasetFactory

import torch 
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from PIL import Image



transform_img = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),    
    transforms.Normalize([0.4355, 0.3777, 0.2879], [0.2571, 0.2028, 0.2101])
])

def tokenize_text(text):
    try:
        return clip.tokenize([text])[0]
    except:
        return clip.tokenize([text.split(".")[0]])[0] 

    
data_path = os.path.join(os.getcwd(), "data") 
dataset = DatasetFactory.Flowers(data_path, transform_img=transform_img, transform_caption=tokenize_text)


t_size = int(0.85 * len(dataset))
v_size = int(0.1 * len(dataset))
t_size = len(dataset) - v_size - t_size

train_sampler = SubsetRandomSampler(list(range(t_size)))
val_sampler = SubsetRandomSampler(list(range(t_size, t_size + v_size)))
test_sampler = SubsetRandomSampler(list(range(t_size + v_size, len(dataset))))

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, pin_memory=True)
val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler, pin_memory=True) 


gan = GAN_INT_CLS()
gan.fit(train_loader, val_loader)
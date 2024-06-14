import os
from torch.utils.data import Dataset
import numpy as np
import torchfile
import torch as torch

from gan_t2i.utils.downloads import download_file, extract_tar_gz, download_file_from_google_drive
from gan_t2i.utils.logger import info, success

class Birds(Dataset):
    
    __DATASET_NAME = "Birds"
    __GOOGLE_ID_CAPTION = "0B0ywwgffWnLLLUc2WHYzM0Q2eWc"
    
    
    def __init__(self, path: str, transform = None, force_download = False):
        super().__init__()
        
        # Path of the dataset 
        d_path = os.path.join(path, self.__DATASET_NAME)
        
        # Captions and Images path in the dataset path
        cap_path = os.path.join(d_path, f"{self.__DATASET_NAME}-caption")
        img_path = os.path.join(d_path, f"{self.__DATASET_NAME}-images")
        
        # Captions and Images tar file
        cap_tar_path = os.path.join(cap_path, f"{self.__DATASET_NAME}-caption.tar.gz")
        img_tar_path = os.path.join(img_path, f"{self.__DATASET_NAME}-images.tar.gz")
        
        # Captions and Images tar extracted path
        cap_ext_path = os.path.join(cap_path, "ext")
        img_ext_path = os.path.join(img_path, "ext")
        
        # Downloading from google drive the caption data
        if force_download or not os.path.exists(cap_path):
            info("Downloading captions ...")
            download_file_from_google_drive(
                file_id=self.__GOOGLE_ID_CAPTION,
                destination=cap_tar_path
            )
            success("\t=> Downloaded captions")
        else:
            info("Captions already downloaded")

        
        # Downloading images data
        #TODO: Download the images
                
        
        # Extract the downloaded captions
        info("Extracting captions ...")
        extract_tar_gz(cap_tar_path, cap_ext_path)
        success("\t=> Extracted captions")
        
        #Extract the downloaded images
        # TODO: Extract the downloaded images
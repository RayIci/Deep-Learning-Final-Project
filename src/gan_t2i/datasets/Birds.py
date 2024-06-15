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
    
    
    # TODO: Finish implementation (search other caption data)
    def __init__(self, path: str, transform_img = None, transform_caption = None, force_download = False, resize = None):
        """ 
        # Birds
        The Birds dataset with images and captions (no classes).
        ## PARAMS
        - path: path of where to download the dataset
        - transform_img: image transformation (applied when retried)
        - transform_caption: caption transformation (applied when retried)
        - force_download: force download
        - resize: resize the images (images are resized instanly and not when retried)
        """
        
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
        
        # Downloading from google drive the caption data creating the folder if not exists
        if force_download or not os.path.exists(cap_path):
            if not os.path.exists(cap_path):
                os.makedirs(cap_path)
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
        
        
        # Extract the downloaded captions creating the ext folder if not exists
        if not os.path.exists(cap_ext_path):
            info("Extracting captions ...")
            os.makedirs(cap_ext_path)
            extract_tar_gz(cap_tar_path, cap_ext_path)
            success("\t=> Extracted captions")

        #Extract the downloaded images
        # TODO: Extract the downloaded images
        
        
        cap_data_path = os.path.join(cap_ext_path, "cub_icml")
        for f_class in os.listdir(cap_data_path):
            
            f_class_cap = os.path.join(cap_data_path, f_class)
            
            for f_image in os.listdir(f_class_cap):
                f_cap_t7 = os.path.join(f_class_cap, f_image)
                loaded_t7 = torchfile.load(f_cap_t7)
                print(loaded_t7)
                # TODO: Load the captions and images (cannot due to t7 error)
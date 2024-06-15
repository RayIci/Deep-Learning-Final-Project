import os
from torch.utils.data import Dataset
import numpy as np
import torch as torch
from PIL import Image
from tqdm import tqdm

from gan_t2i.utils.downloads import download_file, extract_tar_gz, download_file_from_google_drive
from gan_t2i.utils.logger import info, success


class Flowers(Dataset):    
    
    __GOOGLE_ID_CAPTION = "0B0ywwgffWnLLcms2WWJQRFNSWXM"
    __DOWNLOADING_PATH_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    __DATASET_NAME = "Flowers102"
    
    def __init__(self, path: str, transform_img = None, transform_caption = None, force_download = False, resize = None):
        """ 
        # Flowers 102
        The Flower 102 dataset with images and captions (no classes).
        ## PARAMS
        - path: path of where to download the dataset
        - transform_img: image transformation (applied when retried)
        - transform_caption: caption transformation (applied when retried)
        - force_download: force download
        - resize: resize the images (images are resized instanly and not when retried)
        """
        
        super().__init__()
        
        self.transform_img = transform_img
        self.transform_caption = transform_caption
        
        
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
        
        # Downloading the tar from google drive of the caption data
        # Creating the folder if not exists
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

        # Downloading the tar images data creating the folder if not exists
        if force_download or not os.path.exists(img_path):
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            
            info("Downloading images ...")
            download_file(
                url=self.__DOWNLOADING_PATH_IMAGES,
                destination=img_tar_path
            )
            success("\t=> Downloaded images")
        else:
            info("images already downloaded")
        
        
        # Extract the downloaded captions creating the ext folder if not exists
        if not os.path.exists(cap_ext_path):
            info("Extracting captions ...")
            os.makedirs(cap_ext_path)
            extract_tar_gz(cap_tar_path, cap_ext_path)
            success("\t=> Extracted captions")
        else:
            info("Captions already extracted")
        
        # Extract the downloaded images creating the ext folder if not exists
        if not os.path.exists(img_ext_path):
            info("Extracting images ...")
            os.makedirs(img_ext_path)
            extract_tar_gz(img_tar_path, img_ext_path)
            success("\t=> Extracted images")
        else:
            info("images already extracted")
            
        
        info("Reading dataset ...")
        # Load the data  TODO: Change in a optimized data structure
        # The final result is a list of tuple in the data variable with (image, caption)
        self.data = np.array([])
        
        # Path of images
        def get_img_path(img_name):
            return os.path.join(img_ext_path, "jpg", img_name)

        # Iterate over the captions in the txt files
        cap_text_path = os.path.join(cap_ext_path, "text_c10")
        for f_class in tqdm(os.listdir(cap_text_path)):
            
            f_class_cap = os.path.join(cap_text_path, f_class)
            if not os.path.isdir(f_class_cap):
                continue
            
            for f_in_class in os.listdir(f_class_cap):
                
                # Read caption from the txt files
                if not os.path.isdir(f_in_class) and f_in_class.endswith(".txt"):
                    f_img_text = os.path.join(f_class_cap, f_in_class)
                    
                    img_name = f"{f_img_text.split('\\')[-1].split('.')[0]}.jpg"
                    img_path = get_img_path(img_name)
                    img = np.array(Image.open(img_path) if resize is None 
                            else Image.open(img_path).resize(resize))

                    # Each line of the txt file is a caption for that img
                    with open(f_img_text, 'r') as f:
                        captions = f.read().splitlines()
                        for caption in captions:
                            self.data = np.append(self.data, {
                                "img": img,
                                "cap": caption
                            })
                            
        success("\t=> Dataset loaded")         
                        
                        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        """ 
        Return the image and the caption for the index (return img, caption)
        """
        img = self.data[index]["img"]
        caption = self.data[index]["cap"]
        
        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_caption is not None:
            caption = self.transform_caption(caption)
        
        return img, caption
import os
from torch.utils.data import Dataset
import numpy as np
import torchfile

from gan_t2i.utils.downloads import download_file, extract_tar_gz, download_file_from_google_drive
from gan_t2i.utils.logger import info, success


class Flowers(Dataset):    

    __GOOGLE_DRIVE_FILE_ID_CAPTION_DATA = "0B0ywwgffWnLLMl9uOU91MV80cVU"
    __DOWNLOADING_PATH_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    __DATASET_NAME = "Flowers102"
    
    def __download_dataset(self):
        """ 
            Refactored function used to download the dataset from the internet
        """
        # ****************************
        # CAPTION DATA DOWNLOAD
        # ****************************
        if self.__force_download or not os.path.exists(self.__CAPTION_DATA_PATH):
            # Defining caption tar file path (to download)
            tar_file_caption = os.path.join(self.__CAPTION_DATA_PATH, f"{self.__DATASET_NAME}-caption.tar.gz")
            
            info("\n\nDownloading flowers caption data ...")
            
            # Check if the caption path exists otherwise create it
            if not os.path.exists(self.__CAPTION_DATA_PATH):
                os.makedirs(self.__CAPTION_DATA_PATH)
                
            # Downloading from google drive the data
            download_file_from_google_drive(
                file_id=self.__GOOGLE_DRIVE_FILE_ID_CAPTION_DATA,
                destination=tar_file_caption
            )
            success("\t=>Downloaded flowers caption data")
            
            # Extract the downloaded tar gz
            ext_path_caption = os.path.join(self.__CAPTION_DATA_PATH, "ext")
            info("Extracting flowers caption data ...")
            extract_tar_gz(tar_file_caption, ext_path_caption)
            success("\t=>Extracted flowers caption data")
        else:
            info("\n\nFlowers caption data already downloaded")
            
            
        # ****************************
        # IMAGES DOWNLOAD
        # ****************************
        if self.__force_download or not os.path.exists(self.__IMAGES_PATH):
            
            # Defining image tar file path (to download)
            tar_file_imag = os.path.join(self.__IMAGES_PATH, f"{self.__DATASET_NAME}-imags.tgz")
            
            info("\n\nDownloading flowers data images ...")
            
            # Check if the images path exists otherwise create it
            if not os.path.exists(self.__IMAGES_PATH):
                os.makedirs(self.__IMAGES_PATH)
            
            # Downloading images data
            download_file(
                url=self.__DOWNLOADING_PATH_IMAGES,
                destination=tar_file_imag,
            )
            success("\t=>Downloaded flowers data images")
            
            # Extract the downloaded tar gz
            ext_path_img = os.path.join(self.__IMAGES_PATH, "ext")
            info("Extracting flowers data images ...")
            extract_tar_gz(tar_file_imag, ext_path_img)
            success("\t=>Extracted flowers data images")
        else:
            info("\n\nFlowers images data already downloaded")
        
    
    def __init__(self, path: str, transform = None, force_download = False):
        super(Flowers, self).__init__()
        
        self.transform = transform
        self.__force_download = force_download
        
        # Setting all the paths 
        self.__DOWNLOADING_PATH = os.path.join(path, self.__DATASET_NAME)
        self.__CAPTION_DATA_PATH = os.path.join(self.__DOWNLOADING_PATH, f"{self.__DATASET_NAME}-caption")
        self.__IMAGES_PATH = os.path.join(self.__DOWNLOADING_PATH, f"{self.__DATASET_NAME}-images")
            
            
        self.__download_dataset()
        
        self.__CAP_FOLDER = os.path.join(self.__CAPTION_DATA_PATH, "ext/flowers_icml/")    
        self.__IMAGES_FOLDER = os.path.join(self.__IMAGES_PATH, "ext/jpg/")

        
        self.images = np.array([])
        self.labels = np.array([])
        
        for f_class in os.listdir(self.__CAP_FOLDER):
            if f_class.startswith("class_"):
                class_folder = os.path.join(self.__CAP_FOLDER, f_class)

                for f_timg in os.listdir(class_folder):
                    timg_file = os.path.join(class_folder, f_timg)
                    print(timg_file)
                    tens_file = torchfile.load(timg_file)
                    #TODO: FIXING LOADING OF TENSORS
                    
    
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    
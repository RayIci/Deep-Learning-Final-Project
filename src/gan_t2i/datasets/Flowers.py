import os
from torch.utils.data import Dataset

from gan_t2i.utils.downloads import download_file, download_file_from_google_drive
from gan_t2i.utils.logger import info, success


class Flowers(Dataset):    
    
    __GOOGLE_DRIVE_ID_DESCRIPTIONS = "0B0ywwgffWnLLZUt0UmQ1LU1oWlU"
    __DOWNLOADING_PATH_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    __DATASET_NAME = "Flowers102"
    
    
    def __init__(self, path: str, transform = None, force_download = False):
        super(Flowers, self).__init__()
        
        # Setting all the paths 
        self.__DOWNLOADING_PATH = os.path.join(path, self.__DATASET_NAME)
        self.__DESCRIPTIONS_PATH = os.path.join(self.__DOWNLOADING_PATH, f"{self.__DATASET_NAME}-descriptions")
        self.__IMAGES_PATH = os.path.join(self.__DOWNLOADING_PATH, f"{self.__DATASET_NAME}-images")
        
        
        # Downloading functions
        def download_descriptions():
            info("\n\nDownloading flowers dataset descriptions ...")
            if not os.path.exists(self.__DESCRIPTIONS_PATH):
                os.makedirs(self.__DESCRIPTIONS_PATH)
            download_file_from_google_drive(
                id=self.__GOOGLE_DRIVE_ID_DESCRIPTIONS, 
                destination=os.path.join(self.__DESCRIPTIONS_PATH, f"{self.__DATASET_NAME}-desc.tar.gz"),
            )
            success("\t=>Downloaded flowers dataset descriptions")
            
        def download_images():
            info("\n\nDownloading flowers dataset images ...")
            if not os.path.exists(self.__IMAGES_PATH):
                os.makedirs(self.__IMAGES_PATH)
            download_file(
                url=self.__DOWNLOADING_PATH_IMAGES,
                destination=os.path.join(self.__IMAGES_PATH, f"{self.__DATASET_NAME}-imags.tgz"),
            )
            success("\t=>Downloaded flowers dataset images")
        
        
        # Actual dataset downlaoding
        if force_download or not os.path.exists(self.__DESCRIPTIONS_PATH):
            download_descriptions()
        else:
            info("\n\nFlowers dataset descriptions already downloaded")
        
        if force_download or not os.path.exists(self.__IMAGES_PATH):
            download_images()
        else:
            info("\n\nFlowers dataset images already downloaded")
        
        #TODO: IMPLEMENT EXTRACTION AND DATASET LOADING
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    
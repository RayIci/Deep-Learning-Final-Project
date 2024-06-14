import os
from torch.utils.data import Dataset

from gan_t2i.utils.downloads import download_file, extract_tar_gz, download_file_from_google_drive
from gan_t2i.utils.logger import info, success


class Flowers(Dataset):    
    
    __GOOGLE_DRIVE_FILE_ID_CAPTION_DATA = "0B0ywwgffWnLLMl9uOU91MV80cVU"
    __DOWNLOADING_PATH_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    __DATASET_NAME = "Flowers102"
    
    def __init__(self, path: str, transform = None, force_download = False):
        super(Flowers, self).__init__()
        
        # Setting all the paths 
        self.__DOWNLOADING_PATH = os.path.join(path, self.__DATASET_NAME)
        self.__CAPTION_DATA_PATH = os.path.join(self.__DOWNLOADING_PATH, f"{self.__DATASET_NAME}-caption")
        self.__IMAGES_PATH = os.path.join(self.__DOWNLOADING_PATH, f"{self.__DATASET_NAME}-images")
        
        
        # Downloading functions
        def download_caption_data():
            tar_file = os.path.join(self.__CAPTION_DATA_PATH, f"{self.__DATASET_NAME}-caption.tar.gz")
            info("\n\nDownloading flowers caption data ...")
            if not os.path.exists(self.__CAPTION_DATA_PATH):
                os.makedirs(self.__CAPTION_DATA_PATH)
            download_file_from_google_drive(
                file_id=self.__GOOGLE_DRIVE_FILE_ID_CAPTION_DATA,
                destination=tar_file
            )
            success("\t=>Downloaded flowers caption data")
            
            ext_path = os.path.join(self.__CAPTION_DATA_PATH, "ext")
            info("Extracting flowers caption data ...")
            print(tar_file)
            extract_tar_gz(tar_file, ext_path)
            success("\t=>Extracted flowers caption data")
            
        def download_images():
            tar_file = os.path.join(self.__IMAGES_PATH, f"{self.__DATASET_NAME}-imags.tgz")
            info("\n\nDownloading flowers data images ...")
            if not os.path.exists(self.__IMAGES_PATH):
                os.makedirs(self.__IMAGES_PATH)
            download_file(
                url=self.__DOWNLOADING_PATH_IMAGES,
                destination=tar_file,
            )
            success("\t=>Downloaded flowers data images")
            
            ext_path = os.path.join(self.__IMAGES_PATH, "ext")
            info("Extracting flowers data images ...")
            extract_tar_gz(tar_file, ext_path)
            success("\t=>Extracted flowers data images")
        
        # Actual dataset downlaoding
        if force_download or not os.path.exists(self.__CAPTION_DATA_PATH):
            download_caption_data()
        else:
            info("\n\nFlowers caption data already downloaded")
        
        if force_download or not os.path.exists(self.__IMAGES_PATH):
            download_images()
        else:
            info("\n\nFlowers images data already downloaded")
        
        #TODO: IMPLEMENT EXTRACTION AND DATASET LOADING
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    
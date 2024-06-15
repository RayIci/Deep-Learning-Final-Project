import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm

from gan_t2i.utils.downloads import download_file, extract_tar_gz, download_file_from_google_drive
from gan_t2i.utils.logger import info, success

class Birds(Dataset):
    
    __DATASET_NAME = "Birds"
    __GOOGLE_ID_CAPTION = "0B0ywwgffWnLLZW9uVHNjb2JmNlE"
    __DOWNLOADING_PATH_IMAGES = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    
    def __init__(self, path: str, transform_img = None, transform_caption = None, force_download = False, resize = None):
        """ 
        # Birds
        The Birds dataset with images and captions (no classes).
        ## PARAMS
        - path: path of where to download the dataset
        - transform_img: image transformation (applied when retried)
        - transform_caption: caption transformation (applied when retried)
        - force_download: force download
        - resize: resize the images (images are resized instanly and not when retrived)
        
        ### SMALL NOTE
        This dataset contains some gray level images. For this reason, all the images that are not in the
        r g b format will be discarded and those will not be added to the dataset.
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

        
        # Dowloading the image data creating the folder if not exists
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
            info("Images already downloaded")
        
        
        # Extract the downloaded captions creating the ext folder if not exists
        if not os.path.exists(cap_ext_path):
            info("Extracting captions ...")
            os.makedirs(cap_ext_path)
            extract_tar_gz(cap_tar_path, cap_ext_path)
            success("\t=> Extracted captions")

        # Extract the downloaded images
        if not os.path.exists(img_ext_path):
            info("Extracting images ...")
            os.makedirs(img_ext_path)
            extract_tar_gz(img_tar_path, img_ext_path)
            success("\t=> Extracted images")
        
        
        info("Reading dataset ...")
        # TODO: Change in a optimized data structure
        # Load the dataset
        # The final result is a list of tuple in the data variable with (image, caption)
        self.data = np.array([])
        
        # Path of images
        def get_img_path(class_number, img_name):
            img_folder = os.path.join(img_ext_path, "CUB_200_2011", "images")
            for folder in os.listdir(img_folder):
                if class_number in folder:
                    return os.path.join(img_folder, folder, img_name)
        
        
        cap_text_path = os.path.join(cap_ext_path, "text_c10")
        for f_class in  tqdm(os.listdir(cap_text_path)):
            class_number = f_class.split('.')[0]
            
            if not os.path.isdir(os.path.join(cap_text_path, f_class)):
                continue
            
            for txt_cap in os.listdir(os.path.join(cap_text_path, f_class)):
                if txt_cap.endswith(".txt"):
                    curr_img_path = get_img_path(class_number, txt_cap.split(".")[0] + ".jpg")
                    img = np.array(Image.open(curr_img_path) if resize is None 
                            else Image.open(curr_img_path).resize(resize))

                    # The dataset have some gray level images
                    # Discard those that are not r g b
                    if len(img.shape) != 3 or img.shape[2] != 3:
                        continue
                    
                    # Each line of the txt file is a caption for that img
                    with open(os.path.join(cap_text_path, f_class, txt_cap), 'r') as f:
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
        
           
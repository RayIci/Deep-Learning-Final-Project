import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import h5py

from gan_t2i.datasets.DATASET_COFIGS import STORING_MODE
from gan_t2i.utils.downloads import download_file, extract_tar_gz, download_file_from_google_drive
from gan_t2i.utils.logger import info, success


class Flowers(Dataset):    
    
    __GOOGLE_ID_CAPTION = "0B0ywwgffWnLLcms2WWJQRFNSWXM"
    __DOWNLOADING_PATH_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    __DATASET_NAME = "Flowers102"
    
    def __init__(self, path: str, storing_mode: STORING_MODE = STORING_MODE.HDF5, resize = (64, 64), transform_img = None, transform_caption = None, transform_class = None, force_download = False):
        """ 
        # Flowers 102
        The Flower 102 dataset with images and captions (no classes).
        ## PARAMS
        - path: path of where to download the dataset
        - storing_mode: the mode in which the dataset is stored
        - resize: resize the image if the storing_mode is HDF5 since it requires a fixed image size
        it must be a size without the channels (the channels are 3: r g b), for example (64, 64) or (24, 24), etc...
        - transform_img: image transformation
        - transform_caption: caption transformation
        - transform_class: class transformation
        - force_download: force download
        """
        
        super().__init__()
        
        self.storing_mode = storing_mode
        self.transform_img = transform_img
        self.transform_caption = transform_caption
        self.transform_class = transform_class
        
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
            
        
        # Check if the dataset is already stored in HDF5 format, if not initialize 
        # the data in HDF5 format if the mode is HDF5.
        hdf5_file_path = os.path.join(d_path, f"{self.__DATASET_NAME}.h5")
        if self.storing_mode == STORING_MODE.HDF5 and os.path.exists(hdf5_file_path):
            self.hdf5_file = h5py.File(hdf5_file_path, 'r')
            self.images = self.hdf5_file['img']
            self.captions = self.hdf5_file['cap']
            self.classes = self.hdf5_file['class']
            success("The dataset is already stored in HDF5 format")
            return 
        
        elif self.storing_mode == STORING_MODE.HDF5:
            self.hdf5_file = h5py.File(hdf5_file_path, 'w')
            self.images = self.hdf5_file.create_dataset(
                'img',
                (1, resize[0], resize[1], 3),
                maxshape=(None, resize[0], resize[1], 3),
                dtype='uint8'
            )
            
            self.captions = self.hdf5_file.create_dataset(
                'cap',
                (1, 1),
                maxshape=(None, 1),
                dtype=h5py.string_dtype()
            )
            
            self.classes = self.hdf5_file.create_dataset(
                'class',
                (1, 1),
                maxshape=(None, 1),
                dtype=np.int32
            )
        elif self.storing_mode == STORING_MODE.MEMORY:
            self.images = []
            self.captions = []
            self.classes = []
        else:
            raise ValueError("Invalid storing mode. Must be either HDF5 or MEMORY")
        
        
        # Load the dataset
        info("Reading dataset ... modality: " + str(self.storing_mode.name))
        
        def get_img_path(img_name):
            """ return the path of an image from the image name """
            return os.path.join(img_ext_path, "jpg", img_name)

        # Iterate over the captions in the txt files
        count_data = 1
        cap_text_path = os.path.join(cap_ext_path, "text_c10")
        for class_file in tqdm(os.listdir(cap_text_path)):
            
            curr_work_path = os.path.join(cap_text_path, class_file)
            if not os.path.isdir(curr_work_path):
                continue
            
            for txt_file in os.listdir(curr_work_path):
                
                # Read caption from the txt files
                if not os.path.isdir(txt_file) and txt_file.endswith(".txt"):
                    f_img_text = os.path.join(curr_work_path, txt_file)
                    
                    img_name = f"{f_img_text.split('\\')[-1].split('.')[0]}.jpg"
                    img_path = get_img_path(img_name)
                    img = Image.open(img_path)
                    class_number = int(class_file.split("_")[1])
                    
                    # Each line of the txt file is a caption for that img
                    with open(f_img_text, 'r') as f:
                        captions = f.read().splitlines()        
                        for caption in captions:
                
                            # HDF5 mode of storing the data
                            if storing_mode == STORING_MODE.HDF5:    
                                self.images.resize((count_data, resize[0], resize[1], 3))
                                self.captions.resize((count_data, 1))
                                self.classes.resize((count_data, 1))
                                
                                self.images[count_data - 1] = img.resize(resize)
                                self.captions[count_data - 1] = caption
                                self.classes[count_data - 1] = class_number
                                
                            # Memory mode of storing the data
                            else:
                                self.images.append(img)
                                self.captions.append(caption)
                                self.classes.append(class_number)
                            
                            count_data += 1
                            
                            
        if storing_mode == STORING_MODE.HDF5:
            self.hdf5_file.close()
            self.hdf5_file = h5py.File(hdf5_file_path, 'r')
            
            
        success("\t=> Dataset loaded")         
                        
                        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        """ 
        Return the image and the caption for the index (return img, caption)
        """
        img = self.images[index]
        caption = self.captions[index]
        class_number = self.classes[index]
        
        if self.storing_mode == STORING_MODE.HDF5:
            img = Image.fromarray(img)
            caption = caption[0].decode("utf-8") 
            class_number = class_number[0]
        
        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_caption is not None:
            caption = self.transform_caption(caption)
        if self.transform_class is not None:
            class_number = self.transform_class(class_number)
        
        return img, caption, class_number
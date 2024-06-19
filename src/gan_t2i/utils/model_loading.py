import os

from enum import Enum

from gan_t2i.utils.downloads import download_file_from_google_drive, extract_tar_gz
from gan_t2i.utils.logger import info, success

class CLIP_DATASETS(Enum):
    """ 
    The dataset on which the CLIP model has been fine tuned
    """
    FLOWERS = "FLOWERS_102"


def download_CLIP_model(dataset: CLIP_DATASETS, download_path: str = os.path.join(os.getcwd(), "models_weights"), force_dowload = False):
    """ 
    # Download the CLIP model checkpoints of a dataset.
    Download and extract the CLIP model checkpoints of a dataset.
    ## ARGUMENTS
    - dataset: the dataset on which the CLIP model has been fine tuned
    - download_path: path of where to download the model checkpoints
    - force_dowload: force the download 
    ## RETURNS
    - the path of the downloaded model checkpoint
    """
    
    clip_path = os.path.join(download_path, "CLIP", f"CLIP~FT_{dataset.name}")

    tar_file_path = os.path.join(clip_path, f"CLIP~FT_{dataset.name}.tar.gz")
    ext_file_path = os.path.join(clip_path, f"CLIP~FT_{dataset.name}.pt")
    
    if not force_dowload and os.path.exists(tar_file_path) and not os.path.exists(ext_file_path):
        info(f"Extracting CLIP model {dataset.name} from {tar_file_path}")
        extract_tar_gz(file_path=tar_file_path, extract_path=clip_path)
        os.rename(os.path.join(clip_path, "CLIP~FT_flowers", "CLIP~FT_flowers.pt"), ext_file_path)
        success(f"\t=> CLIP model {dataset.name} extracted")
        return ext_file_path
    elif not force_dowload and os.path.exists(ext_file_path):
        success(f"CLIP model {dataset.name} already exits at {ext_file_path}")
        return ext_file_path
    
    if not os.path.exists(clip_path):
        os.makedirs(clip_path)
        
    
    if dataset == CLIP_DATASETS.FLOWERS:
        info(f"Downloading CLIP model {dataset.name}")
        download_file_from_google_drive("1pmmTGykbheEfi6cCXHY2BLysztkPfpEM", destination=tar_file_path)
        success(f"\t=> CLIP model {dataset.name} downloaded")
        info(f"Extracting CLIP model {dataset.name} from {tar_file_path}")
        extract_tar_gz(file_path=tar_file_path, extract_path=clip_path)
        os.rename(os.path.join(clip_path, "CLIP~FT_flowers", "CLIP~FT_flowers.pt"), ext_file_path)
        success(f"\t=> CLIP model {dataset.name} extracted")


    return ext_file_path
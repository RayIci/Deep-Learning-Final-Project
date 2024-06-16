from enum import Enum

class STORING_MODE(Enum):
    """ 
    # MODALITY
    The modality in which the dataset is loaded
    ## Enumerations
    - HDF5: The dataset is loaded in HDF5 format. A HDF5 file is created in the 
    dataset folder with the name `<dataset_name>.hdf5` and it is used to load the data
    at runtime.
    - MEMORY: The dataset is loaded in memory. The dataset is loaded in memory. 
    """
    HDF5 = 1
    MEMORY = 2
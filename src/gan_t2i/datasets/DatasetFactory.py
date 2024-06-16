from gan_t2i.datasets.Birds import Birds
from gan_t2i.datasets.DATASET_COFIGS import STORING_MODE
from gan_t2i.datasets.Flowers import Flowers


class DatasetFactory():
    """ 
    Dataset factory class.
    Currently supports the following datasets:
    - Flowers 102
    - Birds 200
    """
    @staticmethod
    def Flowers(path: str, storing_mode: STORING_MODE = STORING_MODE.HDF5, resize = (64, 64), transform_img = None, transform_caption = None, transform_class = None, force_download = False):
        return Flowers(
            path=path, 
            storing_mode=storing_mode,
            resize=resize,
            transform_img=transform_img, 
            transform_caption=transform_caption, 
            transform_class=transform_class,
            force_download=force_download, 
        )
    
    @staticmethod
    def Birds(path: str, storing_mode: STORING_MODE = STORING_MODE.HDF5, resize = (64, 64), transform_img = None, transform_caption = None, transform_class = None, force_download = False):
        return Birds(
            path=path, 
            storing_mode=storing_mode,
            resize=resize,
            transform_img=transform_img, 
            transform_caption=transform_caption, 
            transform_class=transform_class,
            force_download=force_download, 
        )
from gan_t2i.datasets.Birds import Birds
from gan_t2i.datasets.Flowers import Flowers

class DatasetFactory():
    # TODO: implement the rest of the datasets
    
    @staticmethod
    def Flowers(path: str, transform_img = None, transform_caption = None, force_download = False, resize = None):
        return Flowers(
            path=path, 
            transform_img=transform_img, 
            transform_caption=transform_caption, 
            force_download=force_download, 
            resize=resize
        )
    
    @staticmethod
    def Birds(path: str, transform_img = None, transform_caption = None, force_download = False, resize = None):
        return Birds(
            path=path, 
            transform_img=transform_img, 
            transform_caption=transform_caption, 
            force_download=force_download, 
            resize=resize
        )
from gan_t2i.datasets.Flowers import Flowers


class DatasetFactory():
    # TODO: implement the rest of the datasets
    
    @staticmethod
    def Flowers(path: str, transform = None, force_download = False):
        return Flowers(path=path, transform=transform, force_download=force_download)
    
    @staticmethod
    def Birds():
        raise NotImplemented("Not implemented yet")
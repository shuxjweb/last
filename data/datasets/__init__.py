# encoding: utf-8

from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .last import LaST
from .last_cloth import LaST_Cloth
from .prcc import PRCC
from .celeba import CELEBA
from .dataset_loader import ImageDataset, ImageDatasetMask, ImageDatasetPath
from .dataset_loader import ImageDatasetVisualMask

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'last': LaST,
    'last_cloth': LaST_Cloth,
    'prcc': PRCC,
    'celeba': CELEBA,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

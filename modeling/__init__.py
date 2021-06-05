# encoding: utf-8


from .baseline import Baseline

from .pcb import pcb_p6, pcb_global
from .hpm import HPM
from .mgn import MGN
from .base_map import BaseRes
from .pyramid import Pyramid


def build_model(num_classes=None, model_type='base', nq=25, pretrain_choice=True):
    if model_type == 'base':
        model = Baseline(num_classes, pretrain_choice)
    elif model_type == 'pcb':
        model = pcb_p6(num_classes)
    elif model_type == 'global':
        model = pcb_global(num_classes)
    elif model_type == 'hpm':
        model = HPM(num_classes)
    elif model_type == 'mgn':
        model = MGN(num_classes)
    elif model_type == 'pyramid':
        model = Pyramid(num_classes=num_classes)
    elif model_type == 'base_res':
        model = BaseRes(num_classes)
    else:
        pass
    return model



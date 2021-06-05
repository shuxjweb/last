import torch
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image
import cv2
"""We expect a list `cfg.transform_list`. The types specified in this list 
will be applied sequentially. Each type name corresponds to a function name in 
this file, so you have to implement the function w.r.t. your custom type. 
The function head should be `FUNC_NAME(in_dict, cfg)`, and it should modify `in_dict`
in place.
The transform list allows us to apply optional transforms in any order, while custom
functions allow us to perform sync transformation for images and all labels.
"""


def hflip(in_dict, cfg):
    # Tricky!! random.random() can not reproduce the score of np.random.random(),
    # dropping ~1% for both Market1501 and Duke GlobalPool.
    # if random.random() < 0.5:
    if np.random.random() < 0.5:
        in_dict['img'] = F.hflip(in_dict['img'])
        in_dict['mask'] = F.hflip(in_dict['mask'])


def resize_3d_np_array(maps, resize_h_w, interpolation):    # [9, 24, 8], [24, 8], 0
    """maps: np array with shape [C, H, W], dtype is not restricted"""
    return np.stack([cv2.resize(m, tuple(resize_h_w[::-1]), interpolation=interpolation) for m in maps])


# Resize image using cv2.resize()
def resize(in_dict, cfg):
    in_dict['img'] = Image.fromarray(cv2.resize(np.array(in_dict['img']), (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR))  # [128, 64] -> [384, 128]
    in_dict['mask'] = Image.fromarray(cv2.resize(np.array(in_dict['mask']), (cfg.width_mask, cfg.height_mask), cv2.INTER_NEAREST), mode='L')


def to_tensor(in_dict, mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225]):
    in_dict['img'] = F.to_tensor(in_dict['img'])      # [3, 256, 128]
    in_dict['img'] = F.normalize(in_dict['img'], mean, std)
    in_dict['mask'] = torch.from_numpy(np.array(in_dict['mask'])).long()       # [48, 16]


def to_tensor_mask(in_dict, mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225]):
    in_dict['mask'] = torch.from_numpy(np.array(in_dict['mask'])).long()       # [48, 16]

def transform(in_dict, transform_list, cfg):
    for t in transform_list:
        eval('{}(in_dict, cfg)'.format(t))
    to_tensor(in_dict)
    return in_dict


def transform_mask(in_dict, transform_list, cfg):
    for t in transform_list:
        eval('{}(in_dict, cfg)'.format(t))
    return in_dict


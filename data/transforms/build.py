# encoding: utf-8


import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from .transforms import RandomErasing, RandomSwap
import collections
import sys
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def build_transforms(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def build_transforms_head(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform



def build_transforms_base(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform



########################################################









def build_transforms_eraser(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

def build_transforms_visual(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    transform = T.Compose([
        T.Resize([cfg.height, cfg.width]),
        T.ToTensor(),
    ])

    return transform



def build_transforms_no_erase(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def build_transforms_visual(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225], use_eraser=False):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if use_eraser:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            RandomErasing(probability=1, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            # normalize_transform
        ])

    return transform






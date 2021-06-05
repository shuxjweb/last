# encoding: utf-8

import numpy as np
import torch
import random
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import data.transforms.transform as transform
import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
import cv2
import os


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



def read_image_s(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)       # 12185

    def __getitem__(self, index):
        try:
            img_path, pid, camid = self.dataset[index]
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)       # [3, 256, 128]
        except:
            print(index)

        return img, pid, camid, img_path





###########################################


class ImageDatasetVisualMask(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)       # 12185

    def __getitem__(self, index):
        try:
            img_path, pid, camid, _ = self.dataset[index]
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)       # [3, 256, 128]
        except:
            print(index)

        return img, pid, camid, img_path



class ImageDatasetMask(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg, transforms, transform_list):
        self.dataset = dataset
        self.cfg = cfg
        self.transform = transforms
        self.transform_list = transform_list

    def __len__(self):
        return len(self.dataset)

    def mask2multi(self, mask):
        mask_n = np.array(mask)       # [16, 8]
        h, w = mask_n.shape
        num = 3
        y = np.zeros((num, h, w))     # background, upper, lower
        y[0][mask_n == 0] = 1
        y[1][(mask_n == 1) | (mask_n == 2) | (mask_n == 3) | (mask_n == 4)] = 1
        y[2][(mask_n == 5) | (mask_n == 6) | (mask_n == 7)] = 1

        vis = np.zeros((num,))
        vis[0] = float(y[0].sum() != 0)
        vis[1] = float(y[1].sum() != 0)
        vis[2] = float(y[2].sum() != 0)

        y = torch.from_numpy(y).float()       # [3, 16, 8]
        vis = torch.from_numpy(vis).float()
        return y, vis


    def get_mask_path(self, img_path):
        img_path_list = img_path.split('/')
        file = img_path_list[-1].split('.')[0] + '.png'
        msk_path = '/'
        for item in img_path_list[:-2]:
            if len(item) == 0:
                continue
            msk_path += item + '/'
        msk_path += 'mask/' + img_path_list[-2] + '/' + file
        return msk_path

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        msk_path = self.get_mask_path(img_path)
        mask = read_image_s(msk_path)

        in_dict = {'img': img, 'mask': mask}
        transform.transform_mask(in_dict, self.transform_list, self.cfg)  # [3, 256, 128], [16, 8]
        img = in_dict['img']                 # [3, 256, 128]
        mask = in_dict['mask']               # [16, 8]

        img = self.transform(img)            # [3, 256, 128]
        mask, vis = self.mask2multi(mask)    # [3, 256, 128], [3,]

        return img, pid, camid, img_path, mask, vis




class ImageDatasetPath(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg, transform_list=None):
        self.dataset = dataset
        self.cfg = cfg
        self.transform_list = transform_list

    def __len__(self):
        return len(self.dataset)

    def get_mask_path(self, img_path):
        img_path_list = img_path.split('/')
        file = img_path_list[-1].split('.')[0] + '.png'
        msk_path = '/'
        for item in img_path_list[:-2]:
            if len(item) == 0:
                continue
            msk_path += item + '/'
        msk_path += 'mask/' + img_path_list[-2] + '/' + file
        return msk_path

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        # msk_path, _, _ = self.data_mask[index]
        msk_path = self.get_mask_path(img_path)
        mask = read_image_s(msk_path)

        in_dict = {'img': img, 'mask': mask}
        transform.transform(in_dict, self.transform_list, self.cfg)  # [3, 256, 128], [16, 8]
        img = in_dict['img']          # [3, 256, 128]
        mask = in_dict['mask']        # [16, 8]

        return img, pid, camid, img_path, mask, img_path





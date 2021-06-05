# encoding: utf-8

import os
import torch
import torch.nn as nn
import numpy as np
from utils.reid_metric import R1_mAP
import shutil
import torch.nn.functional as F
from collections import OrderedDict


def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)
    return img_flip.cuda()


def norm(f):
    # f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def inference(model, test_loader, num_query, return_f=False):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    features = OrderedDict()
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):        # len(test_loader)=151
            data, pid, cmp, fnames = batch              # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            f1 = model(data)  # [128, 3840]
            f2 = model(fliplr(data))  # [128, 3840]
            f = 0.5 * (f1 + f2)  # [128, 2048]
            f = norm(f)  # [128, 2048]
            metric.update([f, pid, cmp])
            if return_f:
                for fname, output in zip(fnames, f):
                    features[fname] = output  # [2048,]
        cmc, mAP = metric.compute()
        if return_f:
            return mAP, cmc[0], cmc[4], cmc[9], cmc[19], features
        else:
            return mAP, cmc[0], cmc[4], cmc[9], cmc[19]


def inference_movie_aligned(model, test_loader, num_query):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):        # len(test_loader)=151
            data, pid, cmp, path = batch              # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            feat, feat_l = model(data)                  # [128, 3840]
            metric.update([feat, pid, cmp])
    cmc, mAP = metric.compute()
    return mAP, cmc[0], cmc[4], cmc[9], cmc[19]


def inference_movie(model, test_loader, num_query):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):        # len(test_loader)=151
            data, pid, cmp, path = batch              # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            feat = model(data)                  # [128, 3840]
            metric.update([feat, pid, cmp])
    cmc, mAP = metric.compute()
    return mAP, cmc[0], cmc[4], cmc[9], cmc[19]

def inference_prcc_global(model, test_loader, num_query, use_flip=True):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            data, pid, cmp, fnames = batch                          # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)                              # [64, 4*2048]

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f

            feat = F.normalize(feat, p=2, dim=-1)

            metric.update([feat, pid, cmp])
    cmc, mAP = metric.compute()
    return mAP, cmc[0], cmc[4], cmc[9], cmc[19]


def inference_base(model, test_loader, num_query):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):        # len(test_loader)=151
            data, pid, cmp = batch              # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            f1 = model(data)  # [128, 3840]
            f2 = model(fliplr(data))  # [128, 3840]
            f = 0.5 * (f1 + f2)  # [128, 2048]
            f = norm(f)  # [128, 2048]
            metric.update([f, pid, cmp])
        cmc, mAP = metric.compute()
        return mAP, cmc[0], cmc[4], cmc[9], cmc[19]


def inference_path(model, test_loader, num_query, use_flip=True):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            data, pid, cmp, fnames = batch                          # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)                              # [64, 4*2048]

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f

            feat = F.normalize(feat, p=2, dim=-1)

            metric.update([feat, pid, cmp])
    cmc, mAP = metric.compute()
    return mAP, cmc[0], cmc[4], cmc[9], cmc[19]






def label2onehot(labels, dim):  # [32,], 6
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)  # 32
    out = torch.zeros(batch_size, dim)  # [32, 6]
    out[np.arange(batch_size), labels.long()] = 1
    return out

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray




def fliplr(img):          # [128, 3, 256, 128]
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)        # [128, 3, 256, 128]
    return img_flip.cuda()




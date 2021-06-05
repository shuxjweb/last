# encoding: utf-8

import torch


def train_collate_fn(batch):
    imgs, pids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def train_collate_fn_path(batch):
    imgs, pids, _, pathes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, pathes

def val_collate_fn_path(batch):
    imgs, pids, camids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths






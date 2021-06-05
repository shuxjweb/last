# encoding: utf-8


import torch
from solver.ranger import Ranger

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params, momentum=cfg.momentum)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params)
    return optimizer


def make_optimizer_map(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params, momentum=cfg.momentum)
    elif cfg.optimizer_name == 'Ranger':
        optimizer = Ranger(params)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params)

    return optimizer




def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params, momentum=cfg.momentum)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.lr_center)
    return optimizer, optimizer_center



def make_optimizer_with_triplet(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params, momentum=cfg.momentum)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params)
    return optimizer



def make_optimizer_with_pcb(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params, momentum=cfg.momentum)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params)

    return optimizer

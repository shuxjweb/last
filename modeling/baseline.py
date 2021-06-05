# encoding: utf-8


import os
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride=1, pretrain_choice=True):
        super(Baseline, self).__init__()
        net = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])

        if pretrain_choice == True:
            net.load_param(r'imagenet/resnet50-19c8e357.pth')
            print('Loading pretrained ImageNet model......')

        net.layer4[0].downsample[0].stride = (1, 1)
        net.layer4[0].conv2.stride = (1, 1)

        self.base = nn.Sequential(*[net.conv1, net.bn1, net.relu])    # [64, 64, 128, 64]
        self.pool = net.maxpool        # [64, 64, 64, 32]
        self.layer1 = net.layer1       # [64, 256, 64, 32]
        self.layer2 = net.layer2       # [64, 512, 32, 16]
        self.layer3 = net.layer3       # [64, 1024, 16, 8]
        self.layer4 = net.layer4       # [64, 2048, 16, 8]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes         # 751

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)       # 2048 -> 751
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, use_cam=False):          # [64, 3, 256, 128]
        t = self.base(x)           # [64, 64, 128, 64]
        t = self.pool(t)           # [64, 64, 64,  32]
        t = self.layer1(t)         # [64, 256, 64, 32]
        t = self.layer2(t)         # [64, 512, 32, 16]
        t = self.layer3(t)         # [64, 1024, 16, 8]
        t = self.layer4(t)         # [64, 2048, 16, 8]

        t_p = self.gap(t)          # [64, 2048, 1, 1]
        t_p = t_p.view(t_p.shape[0], -1)        # [64, 2048]

        t_b = self.bottleneck(t_p)              # [64, 2048]

        if self.training:
            prob = self.classifier(t_b)         # [64, 751]
            return prob, t_p
        else:
            return t_b


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])




def load_checkpoint(fpath):
    if os.path.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    tgt_state_m = list(tgt_state)[0]
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        new_name = name
        if 'module' in name and 'module' not in tgt_state_m:
            new_name = new_name[7:]
        if new_name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[new_name].size():
            print('mismatch:', name, param.size(), tgt_state[new_name].size())
            continue
        tgt_state[new_name].copy_(param)
        copied_names.add(new_name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from modeling.res_net import resnet50
# from random_erasing import RandomErasing_vertical, RandomErasing_2x2
import math

__all__ = ['HPM']


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)


def pcb_block(num_ftrs, num_stripes, local_conv_out_channels, num_classes, avg=False):
    if avg:
        pooling_list = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_stripes)])
    else:
        pooling_list = nn.ModuleList([nn.AdaptiveMaxPool2d(1) for _ in range(num_stripes)])
    conv_list = nn.ModuleList([nn.Conv2d(num_ftrs, local_conv_out_channels, 1, bias=False) for _ in range(num_stripes)])
    batchnorm_list = nn.ModuleList([nn.BatchNorm2d(local_conv_out_channels) for _ in range(num_stripes)])
    relu_list = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(num_stripes)])
    fc_list = nn.ModuleList([nn.Linear(local_conv_out_channels, num_classes, bias=False) for _ in range(num_stripes)])
    for m in conv_list:
        weight_init(m)
    for m in batchnorm_list:
        weight_init(m)
    for m in fc_list:
        weight_init(m)
    return pooling_list, conv_list, batchnorm_list, relu_list, fc_list


def spp_vertical(feats, pool_list, conv_list, bn_list, relu_list, fc_list, num_strides, feat_list=[], logits_list=[]):      # [64, 2048, 24, 8]
    for i in range(num_strides):                # 2
        pcb_feat = pool_list[i](feats[:, :, i * int(feats.size(2) / num_strides): (i + 1) * int(feats.size(2) / num_strides), :])   # [64, 2048, 1, 1]
        pcb_feat = conv_list[i](pcb_feat)       # [64, 256, 1, 1]
        pcb_feat = bn_list[i](pcb_feat)         # [64, 256, 1, 1]
        pcb_feat = relu_list[i](pcb_feat)       # [64, 256, 1, 1]
        pcb_feat = pcb_feat.view(pcb_feat.size(0), -1)       # [64, 256]
        feat_list.append(pcb_feat)              # [64, 256]
        logits_list.append(fc_list[i](pcb_feat))             # [64, 751]
    return feat_list, logits_list       # [64, 256], [64, 751]


def global_pcb(feats, pool, conv, bn, relu, fc, feat_list=[], logits_list=[]):         # [64, 2048, 24, 8]
    global_feat = pool(feats)           # [64, 2048, 1, 1]
    global_feat = conv(global_feat)     # [64, 256, 1, 1]
    global_feat = bn(global_feat)
    global_feat = relu(global_feat)     # [64, 256, 1, 1]
    global_feat = global_feat.view(feats.size(0), -1)     # [64, 256]
    feat_list.append(global_feat)
    logits_list.append(fc(global_feat))     # [64, 751]
    return feat_list, logits_list           # [64, 256], [64, 751]


def global_cam(feats, conv, fc):         # [64, 2048, 24, 8]
    t_cam = feats.detach().clone().requires_grad_(True)  # [64, 2048, 16, 8]
    t_cam = conv(t_cam)  # [64, 256, 16, 8]
    cam = F.conv2d(t_cam, fc.weight.unsqueeze(dim=2).unsqueeze(dim=3).detach().clone())
    cam = F.relu(cam)

    return cam




class HPM(nn.Module):
    def __init__(self, num_classes, num_stripes=6, local_conv_out_channels=256, erase=0, loss={'htri'}, avg=False, **kwargs):
        super(HPM, self).__init__()
        self.erase = erase
        self.num_stripes = num_stripes      # 6
        self.loss = loss

        model_ft = resnet50(pretrained=True, last_conv_stride=1)
        self.num_ftrs = list(model_ft.layer4)[-1].conv1.in_channels     # 2048
        self.features = model_ft
        # PSP
        # self.psp_pool, self.psp_conv, self.psp_bn, self.psp_relu, self.psp_upsample, self.conv = psp_block(self.num_ftrs)

        # global
        self.global_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_conv = nn.Conv2d(self.num_ftrs, local_conv_out_channels, 1, bias=False)     # 2048->256
        self.global_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_fc = nn.Linear(local_conv_out_channels, num_classes, bias=False)            # 256->751

        weight_init(self.global_conv)
        weight_init(self.global_bn)
        weight_init(self.global_fc)

        # 2x
        self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list = pcb_block(self.num_ftrs, 2, local_conv_out_channels, num_classes, avg)
        # 4x
        self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list = pcb_block(self.num_ftrs, 4, local_conv_out_channels, num_classes, avg)
        # 8x
        self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list = pcb_block(self.num_ftrs, 8, local_conv_out_channels, num_classes, avg)

    def forward(self, x, use_cam=False):       # [64, 3, 384, 128]
        feats = self.features(x)          # [64, 2048, 24, 8]
        # assert feats.size(2) == 24
        # assert feats.size(-1) == 8
        # assert feats.size(2) % self.num_stripes == 0

        if use_cam:
            cam = global_cam(feats, self.global_conv, self.global_fc)
            return cam

        if self.erase > 0:
            #    print('Random Erasing')
            erasing = RandomErasing_vertical(probability=self.erase)
            feats = erasing(feats)

        feat_list, logits_list = global_pcb(feats, self.global_pooling, self.global_conv, self.global_bn, self.global_relu, self.global_fc, [], [])        # [64, 256], [64, 751]
        feat_list, logits_list = spp_vertical(feats, self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list, 2, feat_list, logits_list)
        feat_list, logits_list = spp_vertical(feats, self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list, 4, feat_list, logits_list)
        feat_list, logits_list = spp_vertical(feats, self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list, 8, feat_list, logits_list)

        if not self.training:
            return torch.cat(feat_list, dim=1)      # [64, 3840]

        if self.loss == {'xent'}:
            return logits_list
        elif self.loss == {'xent', 'htri'}:
            return logits_list, feat_list
        elif self.loss == {'htri'}:
            return logits_list, feat_list       # [64, 751], [64, 256]
        elif self.loss == {'cent'}:
            return logits_list, feat_list
        elif self.loss == {'ring'}:
            return logits_list, feat_list
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

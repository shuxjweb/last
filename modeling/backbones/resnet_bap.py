# encoding: utf-8


import math
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, groups=1,  dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


### Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, **kwargs):
        super(BAP, self).__init__()

    def forward(self, feature_maps, attention_maps):    # [64, 512, 16, 8], [64, 32, 16, 8]
        feature_shape = feature_maps.size()             # [64, 512, 16, 8]
        attention_shape = attention_maps.size()         # [64, 32, 16, 8]
        # print(feature_shape,attention_shape)
        phi_I = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps))      # [64, 32, 512]
        phi_I = torch.div(phi_I, float(attention_shape[2] * attention_shape[3]))    # [64, 32, 512]    min=0, max=0.0543
        phi_I = torch.mul(torch.sign(phi_I), torch.sqrt(torch.abs(phi_I) + 1e-12))  # [64, 32, 512] mul: dot multiply, mm: matrix multiply   min=0, max=0.2329
        phi_I = phi_I.view(feature_shape[0], -1)        # [64, 16384]
        raw_features = torch.nn.functional.normalize(phi_I, dim=-1)                 # [64, 16384]
        pooling_features = raw_features * 100           # [64, 16384]
        # print(pooling_features.shape)
        return raw_features, pooling_features           # [64, 16384], [64, 16384]


class ResizeCat(nn.Module):
    def __init__(self, **kwargs):
        super(ResizeCat, self).__init__()

    def forward(self, at1, at3, at5):
        N, C, H, W = at1.size()
        resized_at3 = nn.functional.interpolate(at3, (H, W))
        resized_at5 = nn.functional.interpolate(at5, (H, W))
        cat_at = torch.cat((at1, resized_at3, resized_at5), dim=1)
        return cat_at


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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_bap=False, parts=32):
        super(Bottleneck, self).__init__()

        ## add by zengh
        self.use_bap = use_bap
        self.parts = parts
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.use_bap:
            self.bap = BAP()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):       # [16, 1024, 16, 8]
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # [64, 2048, 16, 8]
        out = self.conv1(x)     # [64, 512, 16, 8]
        out = self.bn1(out)
        out = self.relu(out)    # [64, 512, 16, 8]
        feature_map = out       # [64, 512, 16, 8]
        out = self.conv2(out)   # [64, 512, 16, 8]
        out = self.bn2(out)
        out = self.relu(out)    # [64, 512, 16, 8]
        if self.use_bap:
            attention = out[:, :self.parts, :, :]  # [64, 512, 16, 8] -> [64, 32, 16, 8]
            raw_features, pooling_features = self.bap(feature_map, attention)   # [64, 16384], [64, 16384]
            return attention, raw_features, pooling_features  # [64, 32, 16, 8], [64, 16384], [64, 16384]
        out = self.conv3(out)   # [64, 2048, 16, 8]
        out = self.bn3(out)

        out += identity
        out = self.relu(out)    # [64, 2048, 16, 8]

        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes=1000, block=Bottleneck, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_bap=False, parts=32):
        super(ResNet, self).__init__()
        self.use_bap = use_bap
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.parts = parts
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2], use_bap=use_bap, parts=parts)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        if use_bap == True:
            self.bottleneck = nn.BatchNorm1d(512 * self.parts)
            self.fc_new = nn.Linear(512 * self.parts, num_classes)
        else:
            self.bottleneck = nn.BatchNorm1d(2048)
            self.fc_new = nn.Linear(2048, num_classes)

        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_bap=False, parts=32):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # if use_bap:
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer, use_bap=use_bap, parts=parts))
        if use_bap:
            return nn.Sequential(*layers)

        for _ in range(2, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):       # [64, 3, 256, 128]
        t = self.conv1(x)       # [64, 64, 128, 64]
        t = self.bn1(t)
        t = self.relu(t)
        t = self.maxpool(t)     # [64, 64, 64, 32]

        t = self.layer1(t)      # [64, 256, 64, 32]
        t = self.layer2(t)      # [64, 512, 32, 16]
        t = self.layer3(t)      # [64, 1024, 16, 8]

        if self.use_bap:
            attention, raw_feat, t = self.layer4(t)          # [64, 32, 16, 8], [64, 16384], [64, 16384]
            prob = self.fc_new(t)                   # [64, 751]
            return attention, raw_feat, prob
        else:
            t = self.layer4(t)                      # [64, 2048, 16, 8]
            t = self.avgpool(t)                     # [64, 2048, 1, 1]
            feat = torch.flatten(t, 1)              # [64, 16384]
            t_b = self.bottleneck(feat)             # [64, 2048]
            prob = self.fc_new(t_b)                 # [64, 751]
            return feat, prob




class ResNetBapNew(nn.Module):
    def __init__(self, layers, num_classes=1000, block=Bottleneck, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_bap=False, parts=8):
        super(ResNetBapNew, self).__init__()
        self.use_bap = use_bap
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.parts = parts
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.use_bap:
            self.bottleneck = nn.BatchNorm1d(2048)
            self.fc_new = nn.Linear(2048 * self.parts, num_classes)

            self.conv2 = conv3x3(2048, self.parts, 1, groups, 1)
            self.bn2 = norm_layer(self.parts)
            self.relu2 = nn.ReLU(inplace=True)
            self.bap = BAP()
        else:
            self.bottleneck = nn.BatchNorm1d(2048)
            self.fc_new = nn.Linear(512 * block.expansion, num_classes)

        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_bap=False, parts=32):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # if use_bap:
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer, use_bap=use_bap, parts=parts))
        if use_bap:
            return nn.Sequential(*layers)

        for _ in range(2, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):       # [64, 3, 256, 128]
        t = self.conv1(x)       # [64, 64, 128, 64]
        t = self.bn1(t)
        t = self.relu(t)
        t = self.maxpool(t)     # [64, 64, 64, 32]

        t = self.layer1(t)      # [64, 256, 64, 32]
        t = self.layer2(t)      # [64, 512, 32, 16]
        t = self.layer3(t)      # [64, 1024, 16, 8]
        t = self.layer4(t)


        if self.use_bap:
            out = self.conv2(t)   # [16, 32, 16, 8]
            out = self.bn2(out)
            attention = self.relu(out)           # [16, 32, 16, 8]
            raw_feat, t = self.bap(t, attention)  # [16, 16384], [16, 16384]
            prob = self.fc_new(t)  # [16, 200]

            return attention, raw_feat, prob  # [16, 32, 14, 14], [16, 16384], [16, 200]
        else:
            t = self.avgpool(t)       # [16, 2048, 1, 1]
            feat = torch.flatten(t, 1)              # [16, 2048]
            t_b = self.bottleneck(feat)  # [64, 2048]
            prob = self.fc_new(t_b)  # [64, 751]
            return feat, prob






import math

import torch
from torch import nn
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):       # [64, 64, 64, 32]
        residual = x

        out = self.conv1(x)     # [64, 64, 64, 32]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)     # [64, 64, 64, 32]
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)     # [64, 256, 64, 32]
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)     # [64, 256, 64, 32]

        out += residual     # [64, 256, 64, 32]
        out = self.relu(out)

        return out



class Model(nn.Module):
    def __init__(self, num_classes, last_stride=1, layers=[3, 4, 6, 3]):
        super(Model, self).__init__()
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)             # [64, 64, 64, 32]
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])                   # [64, 256, 64, 32]
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)        # [64, 512, 32, 16]
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)        # [64, 1024, 16, 8]
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=last_stride)        # [64, 2048, 16, 8]

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes  # 751
        self.dropout = nn.Dropout(p=0.5)

        # self.conv2 = nn.Sequential(*[nn.Conv2d(2048, 512, 1, stride=1, padding=0, bias=False),
        #                              nn.BatchNorm2d(512),
        #                              nn.LeakyReLU(0.1)])
        self.classifier = nn.ModuleList([nn.Linear(2048, num_classes) for _ in range(3)])
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, mask):         # [64, 3, 256, 128], [16, 3, 16, 8]
        t = self.conv1(x)               # [64, 64, 128, 64]
        t = self.bn1(t)                 # [64, 64, 128, 64]
        t = self.relu(t)                # [64, 64, 128, 64]
        t = self.maxpool(t)             # [64, 64, 64,  32]
        t = self.layer1(t)              # [64, 256, 64, 32]
        t = self.layer2(t)              # [64, 512, 32, 16]
        t = self.layer3(t)              # [64, 1024, 16, 8]
        t = self.layer4(t)              # [64, 2048, 16, 8]

        t_p = []
        for ii in range(mask.shape[1]):
            # tt = (t * mask[:, ii:ii+1, :, :]).sum(-1).sum(-1) / mask[:, ii:ii+1, :, :].sum(-1).sum(-1)      # [64, 2048]
            tt = (t * mask[:, ii:ii + 1, :, :]).sum(-1).sum(-1)  # [64, 2048]
            tt = tt.reshape(tt.shape[0], tt.shape[1], 1, 1)      # [64, 2048, 1, 1]
            t_p.append(tt)
        t_p = torch.cat(t_p, dim=2)                              # [64, 2048, 3, 1]
        feat = t_p.permute(0, 2, 1, 3).squeeze(dim=3)            # [64, 3, 2048]

        # t_d = self.conv2(t_d)                                  # [64, 512, 3, 1]

        if self.training:
            t_d = self.dropout(t_p)                              # [64, 512, 3, 1]
            prob = []
            for jj in range(t_d.shape[2]):
                tt = t_d[:, :, jj, :].reshape(t_d.shape[0], -1)  # [64, 2048]
                tt = self.classifier[jj](tt)                     # [64, 751]
                prob.append(tt.unsqueeze(1))
            prob = torch.cat(prob, dim=1)                        # [64, 3, 751]
            return prob, feat.reshape(feat.shape[0], -1)         # [64, 3, 751], [64, 3*2048]
        else:
            feat = F.normalize(feat, p=2, dim=2)
            return feat.reshape(feat.shape[0], -1)               # [64, 3*2048]


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])






class PCB(nn.Module):
    def __init__(self, num_classes, last_stride=1, parts=3, layers=[3, 4, 6, 3]):
        super(PCB, self).__init__()
        self.inplanes = 64
        super().__init__()
        self.parts = parts
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)             # [64, 64, 64, 32]
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])                   # [64, 256, 64, 32]
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)        # [64, 512, 32, 16]
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)        # [64, 1024, 16, 8]
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=last_stride)        # [64, 2048, 16, 8]

        self.parts_pool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes  # 751
        self.dropout = nn.Dropout(p=0.5)

        # self.conv2 = nn.Sequential(*[nn.Conv2d(2048, 512, 1, stride=1, padding=0, bias=False),
        #                              nn.BatchNorm2d(512),
        #                              nn.LeakyReLU(0.1)])
        self.classifier = nn.ModuleList([nn.Linear(2048, num_classes) for _ in range(3)])
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):               # [64, 3, 256, 128], [16, 3, 16, 8]
        t = self.conv1(x)               # [64, 64, 128, 64]
        t = self.bn1(t)                 # [64, 64, 128, 64]
        t = self.relu(t)                # [64, 64, 128, 64]
        t = self.maxpool(t)             # [64, 64, 64,  32]
        t = self.layer1(t)              # [64, 256, 64, 32]
        t = self.layer2(t)              # [64, 512, 32, 16]
        t = self.layer3(t)              # [64, 1024, 16, 8]
        t = self.layer4(t)              # [64, 2048, 16, 8]

        t_p = self.parts_pool(t)        # [64, 2048, 3, 1]
        feat = t_p.permute(0, 2, 1, 3).squeeze(dim=3)            # [64, 3, 2048]

        # t_d = self.conv2(t_d)                                  # [64, 512, 3, 1]

        if self.training:
            t_d = self.dropout(t_p)                              # [64, 512, 3, 1]
            prob = []
            for jj in range(t_d.shape[2]):
                tt = t_d[:, :, jj, :].reshape(t_d.shape[0], -1)  # [64, 2048]
                tt = self.classifier[jj](tt)                     # [64, 751]
                prob.append(tt.unsqueeze(1))
            prob = torch.cat(prob, dim=1)                        # [64, 3, 751]
            return prob, feat.reshape(feat.shape[0], -1)         # [64, 3, 751], [64, 3*2048]
        else:
            feat = F.normalize(feat, p=2, dim=2)
            return feat.reshape(feat.shape[0], -1)               # [64, 3*2048]


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])



############################################################


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None       # [1024,]
        self.bias = None         # [1024,]
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):      # [8, 128, 64, 32]
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)     # 8, 128
        running_mean = self.running_mean.repeat(b).type_as(x)       # [1024,]
        running_var = self.running_var.repeat(b).type_as(x)         # [1024,]
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])   # [1, 1024, 64, 32]
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])      # [8, 128, 64, 32]

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, fp16=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        self.norm = nn.BatchNorm2d(norm_dim)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):    # [64, 2048, 16, 8]
        x = self.conv(self.pad(x))   # [64, 2048, 18, 10] -> [64, 2048, 16, 8]
        if self.norm:
            x = self.norm(x)         # [64, 2048, 16, 8]
        if self.activation:
            x = self.activation(x)   # [64, 2048, 16, 8]
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]

        self.res_type = res_type
        self.model = nn.Sequential(*model)

    def forward(self, x):        # [64, 2048, 16, 8]
        residual = x             # [64, 2048, 16, 8]
        out = self.model(x)      # [64, 2048, 16, 8]
        out += residual          # [64, 2048, 16, 8]
        return out               # [8, 64, 64, 32]

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):      # 4
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):    # [64, 2048, 16, 8]
        return self.model(x)


# class Decoder(nn.Module):
#     def __init__(self, n_upsample, n_res, dim, output_dim, dropout=0, res_norm='bn', activ='relu', pad_type='zero', res_type='basic', fp16=False):
#         super(Decoder, self).__init__()
#         self.input_dim = dim        # 128
#         self.model = []
#         self.model += [nn.Dropout(p=dropout)]
#         self.model += [Conv2dBlock(dim, dim//2, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]   # 2048->1024
#         dim //= 2
#
#         for i in range(n_upsample):     # 4
#             self.model += [nn.Upsample(scale_factor=2),
#                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='bn', activation=activ, pad_type=pad_type, fp16=fp16)]
#             dim //= 2
#         # use reflection padding in the last conv layer
#         self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]      # 32 -> 32
#         self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type)]     # 32 -> 3
#         self.model = nn.Sequential(*self.model)
#
#     def forward(self, x):          # [64, 2048, 16, 8]
#         output = self.model(x)     # [64, 3, 256, 128]
#         return output



###########################################################################
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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = Conv(in_channels, out_channels)

    def forward(self, x):
        t = self.up(x)
        t = self.conv(t)
        return t


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, dim=2048):
        super(Decoder, self).__init__()
        self.input_dim = dim

        self.up1 = Up(in_channels=2048, out_channels=1024)
        self.up2 = Up(in_channels=1024, out_channels=512)
        self.up3 = Up(in_channels=512, out_channels=256)
        self.up4 = Up(in_channels=256, out_channels=128)
        self.conv = OutConv(128, 3)

    def forward(self, x):          # [64, 2048, 16, 8]
        t = self.up1(x)            # [64, 1024, 32, 16]
        t = self.up2(t)            # [64, 512, 64, 32]
        t = self.up3(t)            # [64, 256, 128, 64]
        t = self.up4(t)            # [64, 128, 256, 128]
        t = self.conv(t)           # [64, 3, 256, 128]
        return t







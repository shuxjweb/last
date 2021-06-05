import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck


def make_model(args):
    return MGN(args)


class MGN(nn.Module):
    def __init__(self, num_classes):
        super(MGN, self).__init__()
        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # pool2d = nn.MaxPool2d
        pool2d = nn.AvgPool2d 

        # self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        # self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        # self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        # self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        # self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        self.maxpool_zg_p1 = nn.AdaptiveAvgPool2d(1)
        self.maxpool_zg_p2 = nn.AdaptiveAvgPool2d(1)
        self.maxpool_zg_p3 = nn.AdaptiveAvgPool2d(1)
        self.maxpool_zp2 = nn.AdaptiveAvgPool2d((2, 1))
        self.maxpool_zp3 = nn.AdaptiveAvgPool2d((3, 1))

        feats = 256
        reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())        # 2048->256

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(feats, num_classes)      # 256->751
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):       # [64, 3, 384, 128]

        x = self.backone(x)     # [64, 1024, 24, 8]

        p1 = self.p1(x)         # [64, 2048, 12, 4]
        p2 = self.p2(x)         # [64, 2048, 24, 8]
        p3 = self.p3(x)         # [64, 2048, 24, 8]

        zg_p1 = self.maxpool_zg_p1(p1)        # [64, 2048, 1, 1]
        zg_p2 = self.maxpool_zg_p2(p2)        # [64, 2048, 1, 1]
        zg_p3 = self.maxpool_zg_p3(p3)        # [64, 2048, 1, 1]

        zp2 = self.maxpool_zp2(p2)            # [64, 2048, 2, 1]
        z0_p2 = zp2[:, :, 0:1, :]             # [64, 2048, 1, 1]
        z1_p2 = zp2[:, :, 1:2, :]             # [64, 2048, 1, 1]

        zp3 = self.maxpool_zp3(p3)            # [64, 2048, 3, 1]
        z0_p3 = zp3[:, :, 0:1, :]             # [64, 2048, 1, 1]
        z1_p3 = zp3[:, :, 1:2, :]             # [64, 2048, 1, 1]
        z2_p3 = zp3[:, :, 2:3, :]             # [64, 2048, 1, 1]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)       # [64, 256]
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)       # [64, 256]
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)       # [64, 256]
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)       # [64, 256]
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)       # [64, 256]
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)       # [64, 256]
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)       # [64, 256]
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)       # [64, 256]

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)         # [64, 751]
        l_p2 = self.fc_id_2048_1(fg_p2)         # [64, 751]
        l_p3 = self.fc_id_2048_2(fg_p3)         # [64, 751]

        l0_p2 = self.fc_id_256_1_0(f0_p2)       # [64, 751]
        l1_p2 = self.fc_id_256_1_1(f1_p2)       # [64, 751]
        l0_p3 = self.fc_id_256_2_0(f0_p3)       # [64, 751]
        l1_p3 = self.fc_id_256_2_1(f1_p3)       # [64, 751]
        l2_p3 = self.fc_id_256_2_2(f2_p3)       # [64, 751]

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)        # [64, 2048]

        if not self.training:
            return predict

        return predict, [fg_p1, fg_p2, fg_p3], [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]

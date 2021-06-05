from __future__ import absolute_import

import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes          # 751
        self.feat_dim = feat_dim                # 2048
        self.use_gpu = use_gpu                  # False

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))           # [751, 2048]

    def forward(self, x, labels):         # x->[64, 2048],  labels->[64,]
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)        # 64
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()  # [16, 751]
        distmat.addmm_(1, -2, x, self.centers.t())         # (x - center)^2

        classes = torch.arange(self.num_classes).long()          # [751,]   [0, 1, 2, ..., 750]
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)         # [16, 751]       [[275,275,...275], [153,153,...,153], ...,]
        mask = labels.eq(classes.expand(batch_size, self.num_classes))            # [16, 751]    one_hot

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]             # 2641.08
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)           # [16,]
        loss = dist.mean()               # 2740.9619
        return loss





if __name__ == '__main__':
    use_gpu = False
    center_loss = CenterLoss(use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)             #  to minimize the intra-class distance
    print(loss)






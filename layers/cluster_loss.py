from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


class ClusterLoss(nn.Module):
    def __init__(self, margin=10, use_gpu=True, ordered=True, ids_per_batch=16, imgs_per_id=4):
        super(ClusterLoss, self).__init__()
        self.use_gpu = use_gpu
        self.margin = margin                       # 10
        self.ordered = ordered                     # True
        self.ids_per_batch = ids_per_batch         # 4
        self.imgs_per_id = imgs_per_id             # 4

    def _euclidean_dist(self, x, y):        # x->[1, 2048],  y->[4, 2048]
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)         # 1, 4
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)             # [1, 4]
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()         # [1, 4]
        dist = xx + yy        # [1, 4]
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist           # [1, 4]

    def _cluster_loss(self, features, targets, ordered=True, ids_per_batch=16, imgs_per_id=4):        # features->[16, 2048], targets->[16,]
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             cluster_loss
        """
        if self.use_gpu:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]           # [0, 1, 2, 3]
                else:
                    unique_labels = targets.cpu().unique().cuda()
            else:
                unique_labels = targets.cpu().unique().cuda()
        else:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.unique()
            else:
                unique_labels = targets.unique()

        inter_min_distance = torch.zeros(unique_labels.size(0))         # [4, ]
        intra_max_distance = torch.zeros(unique_labels.size(0))         # [4, ]
        center_features = torch.zeros(unique_labels.size(0), features.size(1))       # [4, 2048]

        if self.use_gpu:
            inter_min_distance = inter_min_distance.cuda()
            intra_max_distance = intra_max_distance.cuda()
            center_features = center_features.cuda()

        index = torch.range(0, unique_labels.size(0) - 1)               # [0, 1, 2, 3]
        for i in range(unique_labels.size(0)):   # the maximum distance among points in the same class
            label = unique_labels[i]
            same_class_features = features[targets == label]            # [4, 2048]
            center_features[i] = same_class_features.mean(dim=0)
            intra_class_distance = self._euclidean_dist(center_features[index == i], same_class_features)       # [1, 4]
            # print('intra_class_distance', intra_class_distance)
            intra_max_distance[i] = intra_class_distance.max()
        # print('intra_max_distance:', intra_max_distance)

        for i in range(unique_labels.size(0)):   # the minmum distance among centers
            inter_class_distance = self._euclidean_dist(center_features[index == i], center_features[index != i])
            # print('inter_class_distance', inter_class_distance)
            inter_min_distance[i] = inter_class_distance.min()
        #  print('inter_min_distance:', inter_min_distance)
        cluster_loss = torch.mean(torch.relu(intra_max_distance - inter_min_distance + self.margin))         # margin=10
        return cluster_loss, intra_max_distance, inter_min_distance

    def forward(self, features, targets):
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             cluster_loss
        """
        assert features.size(0) == targets.size(0), "features.size(0) is not equal to targets.size(0)"
        cluster_loss, cluster_dist_ap, cluster_dist_an = self._cluster_loss(features, targets, self.ordered,
                                                                            self.ids_per_batch, self.imgs_per_id)
        return cluster_loss, cluster_dist_ap, cluster_dist_an


class ClusterLoss_local(nn.Module):
    def __init__(self, margin=10, use_gpu=True, ordered=True, ids_per_batch=32, imgs_per_id=4):
        super(ClusterLoss_local, self).__init__()
        self.use_gpu = use_gpu
        self.margin = margin                   # 10
        self.ordered = ordered                 # True
        self.ids_per_batch = ids_per_batch     # 4
        self.imgs_per_id = imgs_per_id         # 4

    def _euclidean_dist(self, x, y):      # [8, 2048], [32, 2048]
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)       # 8, 32
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)          # [8, 32]
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()      # [8, 32]
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _shortest_dist(self, dist_mat):
        """Parallel version.
        Args:
          dist_mat: pytorch Variable, available shape:
            1) [m, n]
            2) [m, n, N], N is batch size
            3) [m, n, *], * can be arbitrary additional dimensions
        Returns:
          dist: three cases corresponding to `dist_mat`:
            1) scalar
            2) pytorch Variable, with shape [N]
            3) pytorch Variable, with shape [*]
        """
        m, n = dist_mat.size()[:2]  # 8, 8
        # Just offering some reference for accessing intermediate distance.
        dist = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if (i == 0) and (j == 0):
                    dist[i][j] = dist_mat[i, j]
                elif (i == 0) and (j > 0):
                    dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
                elif (i > 0) and (j == 0):
                    dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
                else:
                    dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
        dist = dist[-1][-1]
        return dist

    def _local_dist(self, x, y):      # x->[1, 8, 2048],  y->[4, 8, 2048]
        """
        Args:
          x: pytorch Variable, with shape [M, m, d]
          y: pytorch Variable, with shape [N, n, d]
        Returns:
          dist: pytorch Variable, with shape [M, N]
        """
        M, m, d = x.size()      # 1, 8, 2048
        N, n, d = y.size()      # 4, 8, 2048
        x = x.contiguous().view(M * m, d)      # [8, 2048]
        y = y.contiguous().view(N * n, d)      # [32, 2048]
        # shape [M * m, N * n]
        dist_mat = self._euclidean_dist(x, y)          # [8, 32]
        dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
        # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
        dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)             # [8, 8, 1, 4]
        # shape [M, N]
        dist_mat = self._shortest_dist(dist_mat)
        return dist_mat

    def _cluster_loss(self, features, targets, ordered=True, ids_per_batch=32, imgs_per_id=4):       # features->[16, 8, 2048],   targets->[16,]
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, H, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             cluster_loss
        """
        if self.use_gpu:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]            # [0, 1, 2, 3]
                else:
                    unique_labels = targets.cpu().unique().cuda()
            else:
                unique_labels = targets.cpu().unique().cuda()
        else:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.unique()
            else:
                unique_labels = targets.unique()

        inter_min_distance = torch.zeros(unique_labels.size(0))
        intra_max_distance = torch.zeros(unique_labels.size(0))
        center_features = torch.zeros(unique_labels.size(0), features.size(1), features.size(2))         # [4, 8, 2048]

        if self.use_gpu:
            inter_min_distance = inter_min_distance.cuda()
            intra_max_distance = intra_max_distance.cuda()
            center_features = center_features.cuda()

        index = torch.range(0, unique_labels.size(0) - 1)         # [0, 1, 2, 3]
        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_features = features[targets == label]      # [4, 8, 2048]
            center_features[i] = same_class_features.mean(dim=0)  # [8, 2048]
            intra_class_distance = self._local_dist(center_features[index == i], same_class_features)        # [1, 4]
            # print('intra_class_distance', intra_class_distance)
            intra_max_distance[i] = intra_class_distance.max()
        # print('intra_max_distance:', intra_max_distance)

        for i in range(unique_labels.size(0)):
            inter_class_distance = self._local_dist(center_features[index == i], center_features[index != i])
            # print('inter_class_distance', inter_class_distance)
            inter_min_distance[i] = inter_class_distance.min()
        # print('inter_min_distance:', inter_min_distance)

        cluster_loss = torch.mean(torch.relu(intra_max_distance - inter_min_distance + self.margin))
        return cluster_loss, intra_max_distance, inter_min_distance

    def forward(self, features, targets):        # features->[16, 8, 2048],   targets->[16,]
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, H, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             cluster_loss
        """
        assert features.size(0) == targets.size(0), "features.size(0) is not equal to targets.size(0)"
        cluster_loss, cluster_dist_ap, cluster_dist_an = self._cluster_loss(features, targets, self.ordered,
                                                                            self.ids_per_batch, self.imgs_per_id)
        return cluster_loss, cluster_dist_ap, cluster_dist_an


if __name__ == '__main__':
    use_gpu = True
    cluster_loss = ClusterLoss(use_gpu=use_gpu, ids_per_batch=4, imgs_per_id=4)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).cuda()
    loss = cluster_loss(features, targets)
    print(loss)

    cluster_loss_local = ClusterLoss_local(use_gpu=use_gpu, ids_per_batch=4, imgs_per_id=4)
    features = torch.rand(16, 8, 2048)
    targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    if use_gpu:
        features = torch.rand(16, 8, 2048).cuda()
        targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).cuda()
    loss = cluster_loss_local(features, targets)
    print(loss)

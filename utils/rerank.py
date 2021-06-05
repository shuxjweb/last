from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import numpy as np
import time

import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


def re_ranking(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.1):   # [15094, 2048], [12936, 2048]
    all_num = input_feature.shape[0]             # 12936
    feat = input_feature.astype(np.float16)      # [12936, 2048]

    if lambda_value != 0:    # 0.1
        print('Computing source distance...')
        # import pdb;pdb.set_trace()
        all_num_source = input_feature_source.shape[0]
        sour_tar_dist = np.power(cdist(input_feature, input_feature_source), 2).astype(np.float16)    # [12936, 15094], min=4.273, max=39.66
        sour_tar_dist = 1 - np.exp(-sour_tar_dist)          # min=0.986, max=1.0
        source_dist_vec = np.min(sour_tar_dist, axis=1)     # [12936,]
        source_dist_vec = source_dist_vec / np.max(source_dist_vec)     # [12936,]
        source_dist = np.zeros([all_num, all_num])          # [12936, 12936]
        for i in range(all_num):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del sour_tar_dist
        del source_dist_vec

    print('Computing original distance...')
    original_dist = cdist(feat, feat).astype(np.float16)
    original_dist = np.power(original_dist, 2).astype(np.float16)
    # import pdb;pdb.set_trace()

    del feat
    euclidean_dist = original_dist
    gallery_num = original_dist.shape[0]  # gallery_num=all_num
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.

    print('Starting re_ranking...')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    # original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(all_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    if lambda_value == 0:
        return jaccard_dist
    else:
        final_dist = jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
        return final_dist




def re_ranking_cyc(input_feature, k1=20, k2=6, lambda_value=0.1):   # [15094, 2048], [12936, 2048]
    all_num = input_feature.shape[0]             # 12936
    feat = input_feature.astype(np.float16)      # [12936, 2048]

    print('Computing original distance...')
    original_dist = cdist(feat, feat).astype(np.float16)           # [12936, 12936]
    original_dist = np.power(original_dist, 2).astype(np.float16)
    # import pdb;pdb.set_trace()

    del feat
    euclidean_dist = original_dist
    gallery_num = original_dist.shape[0]  # gallery_num=all_num
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.

    print('Starting re_ranking...')
    for i in range(all_num):        # 12936
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    # original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(all_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist



def ranking_cyc(input_feature):   # [15094, 2048], [12936, 2048]
    print('Computing original distance...')
    original_dist = cdist(input_feature, input_feature).astype(np.float16)           # [12936, 12936]
    original_dist = original_dist / np.max(original_dist, axis=0)

    pos_bool = (original_dist < 0)
    original_dist[pos_bool] = 0.0

    return original_dist





def k_reciprocal_neigh(initial_rank, i, k1):                # [12936, 12936], 0, 21
    forward_k_neigh_index = initial_rank[i, :k1 + 1]        # [21,]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]  # [21, 21]
    fi = torch.nonzero(backward_k_neigh_index == i)[:, 0]   # [16,]
    return forward_k_neigh_index[fi]        # [5,]


def compute_jaccard_dist(target_features, k1=20, k2=1, print_flag=False, lambda_value=0, source_features=None, use_gpu=False):  # [12936, 2048]
    end = time.time()
    N = target_features.size(0)  # 12936
    if (use_gpu):  # False
        # accelerate matrix distance computing
        target_features = target_features.cuda()        # [13605, 2048]
        if (source_features is not None):
            source_features = source_features.cuda()

    if ((lambda_value > 0) and (source_features is not None)):
        M = source_features.size(0)
        sour_tar_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True).expand(N, M) + \
                        torch.pow(source_features, 2).sum(dim=1, keepdim=True).expand(M, N).t()
        sour_tar_dist.addmm_(1, -2, target_features, source_features.t())
        sour_tar_dist = 1 - torch.exp(-sour_tar_dist)
        sour_tar_dist = sour_tar_dist.cpu()
        source_dist_vec = sour_tar_dist.min(1)[0]
        del sour_tar_dist
        source_dist_vec /= source_dist_vec.max()
        source_dist = torch.zeros(N, N)
        for i in range(N):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del source_dist_vec

    if print_flag:
        print('Computing original distance...')

    original_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True) * 2          # [13605, 1]
    original_dist = original_dist.expand(N, N) - 2 * torch.mm(target_features, target_features.t())  # [13605, 13605]
    original_dist /= original_dist.max(0)[0]                # [13605, 13605]
    original_dist = original_dist.t()                       # [13605, 13605] min=-4.49^-7, max=1
    initial_rank = torch.argsort(original_dist, dim=-1)     # [13605, 13605]

    original_dist = original_dist.cpu()                     # [13605, 13605]
    initial_rank = initial_rank.cpu()
    all_num = gallery_num = original_dist.size(0)           # 13605

    del target_features
    if (source_features is not None):
        del source_features

    if print_flag:
        print('Computing Jaccard distance...')

    nn_k1 = []
    nn_k1_half = []
    for i in range(all_num):           # 13605
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = torch.zeros(all_num, all_num)  # [13605, 13605]
    for i in range(all_num):
        k_reciprocal_index = nn_k1[i]  # [16,]
        k_reciprocal_expansion_index = k_reciprocal_index  # [16,]
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]  # [8,]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)   # [21,]  element-wise unique
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])         # [21,]
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)     # softmax, only for right pair samples

    if k2 != 1:  # 6
        k2_rank = initial_rank[:, :k2].clone().view(-1)     # [81630,]
        V_qe = V[k2_rank]           # [81630, 13605]
        V_qe = V_qe.view(initial_rank.size(0), k2, -1).sum(1)       # [13605, 6, 13605] -> [13605, 13605]
        V_qe /= k2                  # [13605, 13605]
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(gallery_num):    # 13605
        invIndex.append(torch.nonzero(V[:, i])[:, 0])   # len(invIndex)=all_num

    jaccard_dist = torch.zeros_like(original_dist)      # [13605, 13605]
    for i in range(all_num):        # 13605
        temp_min = torch.zeros(1, gallery_num)          # [1, 13605]
        indNonZero = torch.nonzero(V[i, :])[:, 0]       # [48,]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):    # 48
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.min(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    del invIndex

    del V

    pos_bool = (jaccard_dist < 0)     # [13605, 13605]
    jaccard_dist[pos_bool] = 0.0      # min=0, max=1
    if print_flag:
        print("Time cost: {}".format(time.time() - end))

    if (lambda_value > 0):
        return jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
    else:
        return jaccard_dist




def compute_base_dist(target_features, use_gpu=False):  # [12936, 2048]
    N = target_features.size(0)  # 12936

    M = target_features.size(0)
    sour_tar_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True).expand(N, M) + \
                    torch.pow(target_features, 2).sum(dim=1, keepdim=True).expand(M, N).t()
    sour_tar_dist.addmm_(1, -2, target_features, target_features.t())
    sour_tar_dist = 1 - torch.exp(-sour_tar_dist)
    jaccard_dist = sour_tar_dist.cpu()

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist





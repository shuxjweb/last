# encoding: utf-8

import numpy as np
import torch

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):     # [3368, 15913], [3368,], [15913,]
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)   # [3368, 15913]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)    # [3368, 15913]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):       # 3368
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)      # [15913,]
        keep = np.invert(remove)      # [15913,]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]       # [15908,]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()          # [15908,]
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()         # 14
        tmp_cmc = orig_cmc.cumsum()      # [15908,], [0,0,0,...,14,14,14]
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]     # [15908,]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc    # [15908,]
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q      # [50,]
    mAP = np.mean(all_AP)

    del indices, matches

    return all_cmc, mAP




def eval_cmc_ap(feat_q, feat_g, q_pid, g_pids, q_camid, g_camids, max_rank=50):     # [3368, 15913], [3368,], [15913,]
    g_pids = g_pids.cpu().data.numpy()
    g_camids = g_camids.cpu().data.numpy()

    if len(feat_q.shape) == 1:
        feat_q = feat_q.unsqueeze(0)        # [1, 2048]

    distmat = torch.pow(feat_q, 2).sum(dim=1, keepdim=True).expand(feat_q.shape[0], feat_g.shape[0]) + \
              torch.pow(feat_g, 2).sum(dim=1, keepdim=True).expand(feat_g.shape[0], feat_q.shape[0]).t()
    distmat.addmm_(1, -2, feat_q, feat_g.t())  # [1, 70264]
    distmat = distmat.cpu().numpy()

    num_q, num_g = distmat.shape            # 1, 70264
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)[0]   # [1, 70264]
    matches = (g_pids[indices] == q_pid).astype(np.int32)    # [70264,]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    # get query pid and camid

    # remove gallery samples that have the same pid and camid with query
    remove = (g_pids[indices] == q_pid) & (g_camids[indices] == q_camid)      # [15913,]
    keep = np.invert(remove)      # [15913,]

    # compute cmc curve
    # binary vector, positions with value 1 are correct matches
    orig_cmc = matches[keep]       # [15908,]
    if not np.any(orig_cmc):
        # this condition is true when query identity does not appear in gallery
        return 0, 0

    cmc = orig_cmc.cumsum()          # [15908,]
    cmc[cmc > 1] = 1

    all_cmc.append(cmc[:max_rank])
    num_valid_q += 1.

    # compute average precision
    # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    num_rel = orig_cmc.sum()         # 14
    tmp_cmc = orig_cmc.cumsum()      # [15908,], [0,0,0,...,14,14,14]
    tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]     # [15908,]
    tmp_cmc = np.asarray(tmp_cmc) * orig_cmc    # [15908,]
    AP = tmp_cmc.sum() / num_rel
    all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q      # [50,]
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_cmc_ap_all(feat_q, feat_g, q_pids, g_pids, q_camids, g_camids, max_rank=50):     # [3368, 15913], [3368,], [15913,]
    q_pids = q_pids.cpu().data.numpy()
    q_camids = q_camids.cpu().data.numpy()
    g_pids = g_pids.cpu().data.numpy()
    g_camids = g_camids.cpu().data.numpy()

    distmat = torch.pow(feat_q, 2).sum(dim=1, keepdim=True).expand(feat_q.shape[0], feat_g.shape[0]) + \
              torch.pow(feat_g, 2).sum(dim=1, keepdim=True).expand(feat_g.shape[0], feat_q.shape[0]).t()
    distmat.addmm_(1, -2, feat_q, feat_g.t())   # [70264, 70264]
    distmat = distmat.cpu().data.numpy()

    num_q, num_g = distmat.shape                # 70264, 70264
    if num_g < max_rank:
        max_rank = num_g
        # print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)       # [1, 70264]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # [3368, 15913]

    # compute cmc curve for each query
    # get query pid and camid
    # remove gallery samples that have the same pid and camid with query
    remove = (g_pids[indices] == q_pids[:, np.newaxis]) & (g_camids[indices] == q_camids[:, np.newaxis])  # [15913,]
    keep = np.invert(remove)  # [7026, 70264]

    # compute cmc curve
    # binary vector, positions with value 1 are correct matches


    # compute average precision
    # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    all_AP = []
    for q_idx in range(num_q):
        orig_cmc = matches[q_idx, keep[q_idx]]     # [15908,]
        num_rel = orig_cmc.sum()      # 14
        tmp_cmc = orig_cmc.cumsum()   # [15908,], [0,0,0,...,14,14,14]
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]  # [15908,]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc        # [15908,]
        ap = tmp_cmc.sum() / num_rel
        all_AP.append(ap)



    return all_AP


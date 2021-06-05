# encoding: utf-8

import argparse
import os
import sys
import torch
import numpy as np
from torch.backends import cudnn
sys.path.append('.')
from data import make_data_loader_movie_ext as make_data_loader
from utils.reid_tool import visualize_ranked_results_all

def load_network_pretrain(model, cfg):
    path = os.path.join(cfg.logs_dir, 'checkpoint.pth')
    if not os.path.exists(path):
        return model, 0, 0.0
    pre_dict = torch.load(path)
    model.load_state_dict(pre_dict['state_dict'])
    start_epoch = pre_dict['epoch']
    best_acc = pre_dict['best_acc']
    print('start_epoch:', start_epoch)
    print('best_acc:', best_acc)
    return model, start_epoch, best_acc


def main(cfg):
    # prepare dataset
    dataset, train_loader, extract_loader, val_loader, test_loader, num_query, num_query_test, num_classes = make_data_loader(cfg)

    # path_f = os.path.join('pre_feat', 'feat_base.npy')
    # feats = np.load(path_f)  # [70264, 2048]

    path_f = os.path.join(cfg.logs_dir, 'feat_test.npy')
    feats = np.load(path_f)  # [70264, 2048]

    feats = torch.from_numpy(feats)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    # query
    qf = feats[:num_query_test]  # [3368, 2048]
    # gallery
    gf = feats[num_query_test:]  # [15913, 2048]
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())  # [3368, 15913]
    distmat = distmat.cpu().numpy()

    save_dir = os.path.join(cfg.logs_dir, 'ranked_results_split')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    visualize_ranked_results_all(distmat, [dataset.query_test,  dataset.gallery_test], save_dir=save_dir, topk=20)

    return


if __name__ == '__main__':
    gpu_id = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="ReID Baseline Training")

    # DATA
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--img_per_id', type=int, default=4)
    parser.add_argument('--batch_size_dis', type=int, default=64)
    parser.add_argument('--img_per_id_dis', type=int, default=8)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height_mask', type=int, default=256)
    parser.add_argument('--width_mask', type=int, default=128)


    # MODEL
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)

    # OPTIMIZER
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--lr_center', type=float, default=0.5)
    parser.add_argument('--center_loss_weight', type=float, default=0.0005)
    parser.add_argument('--steps', type=list, default=[40, 100])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--cluster_margin', type=float, default=0.3)
    parser.add_argument('--bias_lr_factor', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--weight_decay_bias', type=float, default=5e-4)
    parser.add_argument('--range_k', type=float, default=2)
    parser.add_argument('--range_margin', type=float, default=0.3)
    parser.add_argument('--range_alpha', type=float, default=0)
    parser.add_argument('--range_beta', type=float, default=1)
    parser.add_argument('--range_loss_weight', type=float, default=1)
    parser.add_argument('--warmup_factor', type=float, default=0.01)
    parser.add_argument('--warmup_iters', type=float, default=10)
    parser.add_argument('--warmup_method', type=str, default='linear')
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--optimizer_name', type=str, default="Adam")
    parser.add_argument('--momentum', type=float, default=0.9)

    # TRAINER
    parser.add_argument('--max_epochs', type=int, default=120)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--mode', type=str, default='ext', help=['train', 'val', 'test', 'ext'])  # change train or test mode
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--num_works', type=int, default=8)

    # misc
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--dataset', type=str, default='lslt')
    parser.add_argument('--data_dir', type=str, default='/data/shuxj/data/PReID/last_new/last/')
    parser.add_argument('--logs_dir', type=str, default=os.path.join(working_dir, 'logs/20201106_movie_ap_triplet_split_b2048_p8_bin30/'))

    cfg = parser.parse_args()

    main(cfg)

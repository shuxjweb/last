# encoding: utf-8

import os
import torch
import datetime
import shutil
import random
import cv2
import numpy as np
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from engine.inference import inference
from engine.inference import inference_path, inference_base, inference_movie_aligned
from engine.inference import inference_prcc_global

from utils.iotools import AverageMeter
import copy
import errno
import shutil
from PIL import Image

def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)
    return img_flip.cuda()


def norm(f):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def normalize_tensor(x):      # [64, 1, 28, 28]
    map_size = x.size()
    aggregated = x.view(map_size[0], map_size[1], -1)      # [64, 1, 784]
    minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)             # [64, 1, 1]
    maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)             # [64, 1, 1]
    normalized = torch.div(aggregated - minimum, maximum - minimum)      # [64, 1, 784]
    normalized = normalized.view(map_size)      # [64, 1, 28, 28]

    return normalized

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr



def do_train_strong(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, (img, target) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,],  len(train_loader)=980
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                            # [64, 150], [64, 2018]

            loss = loss_fn(scores, feats, target)               # 9.0224
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference_base(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('lr', float(lr), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference_base(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()




def do_train_prcc_global_adversarial_base(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        lr = float(get_lr(optimizer))

        for ii, (img, target, path) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,],  [64,], [64, 6, 256, 128]
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            b, c, h, w = img.shape

            score, feat = model(img)                            # [64, 150], [64, 2018]

            loss = loss_fn(score, feat, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)

            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, lr))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference_path(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': acc_test,
            }, is_best, fpath=cfg.logs_dir)
            last_acc_val = acc_test


        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()


    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference_path(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, cmc5: {:.1%}, cmc10: {:.1%}, cmc20: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()




def do_train_movie_pcb(cfg, model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, num_query, num_query_test, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, input in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            img, target, pathes = input

            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)  # predict, [fg_p1, fg_p2, fg_p3], [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]

            loss = loss_fn(scores, feats, target)
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores.mean(dim=1).max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, val_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': acc_test,
            }, is_best, fpath=cfg.logs_dir)
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query_test)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()





def do_train_movie_base(cfg, model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, num_query, num_query_test, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, input in enumerate(train_loader):       # [64, 3, 256, 128],  [64,],  len(train_loader)=980
            optimizer.zero_grad()

            img, target, path = input
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                # [64, 5000], [64, 2048]

            loss = loss_fn(scores, feats, target)
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, val_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('lr', float(lr), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query_test)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_last_cloth_base(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, input in enumerate(train_loader):       # [64, 3, 256, 128],  [64,],  len(train_loader)=980
            optimizer.zero_grad()

            img, target, path = input
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                # [64, 5000], [64, 2048]

            loss = loss_fn(scores, feats, target)
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('lr', float(lr), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_movie_hpm(cfg, model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, num_query, num_query_test, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, (img, target, path) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                          # [64, 751], [64, 256]

            loss = []
            for jj, (score, feat) in enumerate(zip(scores, feats)):          # [64, 751], [64, 256]
                loss.append(loss_fn(score, feat, target))                 # 11.1710
            # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
            loss = torch.stack(loss).sum()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % 196 == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference_movie_aligned(model, val_loader, num_query)
        acc_test = cmc1
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': acc_test,
            }, is_best, fpath=cfg.logs_dir)
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference_movie_aligned(model, test_loader, num_query_test)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_movie_mgn(cfg, model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, num_query, num_query_test, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, (img, target, path) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            feats, feat_sub, scores = model(img)  # predict, [fg_p1, fg_p2, fg_p3], [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]

            loss = loss_fn(feat_sub, scores, target)
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores[0].max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference_prcc_global(model, val_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': acc_test,
            }, is_best, fpath=cfg.logs_dir)
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference_prcc_global(model, test_loader, num_query_test)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_strong_path(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, input in enumerate(train_loader):               # [64, 3, 256, 128],  [64,],  len(train_loader)=980
            img, target, path = input
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                          # [64, 134], [64, 2048]

            loss = loss_fn(scores, feats, target)               # 9.0224
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('lr', float(lr), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()




def do_train_celeba_global_adversarial_base(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        lr = scheduler.get_last_lr()[0]

        for ii, (img, target, path) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,],  [64,], [64, 6, 256, 128]
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            b, c, h, w = img.shape

            score, feat = model(img)                            # [64, 150], [64, 2018]

            loss = loss_fn(score, feat, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)

            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, lr))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference_path(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()


    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference_path(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, cmc5: {:.1%}, cmc10: {:.1%}, cmc20: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()





def save_checkpoint(state, is_best, fpath):
    if len(fpath) != 0:
        mkdir_if_missing(fpath)

    fpath = os.path.join(fpath, 'checkpoint.pth')
    torch.save(state, fpath, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'checkpoint_best.pth'))



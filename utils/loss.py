import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class LossComputation(nn.Module):
    def __init__(self, num_classes=5000, feature_size=512):
        super(LossComputation, self).__init__()
        self.num_classes = num_classes                    # 12003
        self.feature_size = feature_size        # 256

        self.scale = 28
        self.margin = 0.2

        self.W = Parameter(torch.randn(self.feature_size, self.num_classes), requires_grad=True)        # [256, 12003]
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def instance_loss(self, visual_embed, textual_embed, labels):   # [64, 256], [64, 256], [64,]
        W_norm = F.normalize(self.W, p=2, dim=0)                    # [256, 12003]

        visual_logits = self.scale * torch.matmul(visual_embed, W_norm)          # [64, 12003]
        textual_logits = self.scale * torch.matmul(textual_embed, W_norm)        # [64, 12003]

        criterion = nn.CrossEntropyLoss(reduction='mean')
        v_loss = criterion(input=visual_logits, target=labels)      # 11.2150
        t_loss = criterion(input=textual_logits, target=labels)     # 11.1045
        loss = v_loss + t_loss      # 22.3195

        return loss, v_loss, t_loss

    def image_loss(self, visual_embed, labels):   # [64, 256], [64, 256], [64,]
        W_norm = F.normalize(self.W, p=2, dim=0)                    # [256, 12003]
        visual_logits = self.scale * torch.matmul(visual_embed, W_norm)          # [64, 12003]

        criterion = nn.CrossEntropyLoss(reduction='mean')
        v_loss = criterion(input=visual_logits, target=labels)      # 11.2150

        return v_loss

    def mask_loss(self, seg_feat, masks):       # [320, 6, 48, 16], [64, 5, 48, 16]->min=0, max=5
        mask_logits = seg_feat                  # [320, 6, 48, 16]
        masks = torch.stack(masks, dim=1)       # [5, 64, 48, 16]
        masks = masks.view(-1, masks.size(-2), masks.size(-1))       # [320, 48, 16], min=0, max=5

        mask_loss = F.cross_entropy(mask_logits, masks.long(), reduction='none')           # [320, 48, 16]
        mask_loss = self.num_parts * mask_loss.mean()           # 25.1739
        return mask_loss

    def global_align_loss(self, visual_norm, textual_norm, labels):       # [64, 256], [64, 256], [64,]
        alpha = 0.6
        beta = 0.4
        scale_pos = 10
        scale_neg = 40

        batch_size = labels.size(0)             # 64
        similarity = torch.matmul(visual_norm, textual_norm.t())    # [64, 64],  min=-0.0513, max=0.1054
        labels_ = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())              # [64, 64], True, False

        loss = 0
        for i in range(batch_size):             # 64
            pred = similarity[i]                # [64,]
            label = labels_[i].float()          # [64,]
            pos_inds = torch.nonzero(label == 1, as_tuple=False).squeeze(1)         # [0, 1, 2, 3]
            neg_inds = torch.nonzero(label == 0, as_tuple=False).squeeze(1)
            loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))      # [4,]
            loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))        # [60,]
            loss += loss_pos.sum() + loss_neg.sum()

            pred = similarity[:, i]             # [64,]
            label = labels_[:, i].float()       # [64,]
            pos_inds = torch.nonzero(label == 1, as_tuple=False).squeeze(1)
            neg_inds = torch.nonzero(label == 0, as_tuple=False).squeeze(1)
            loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
            loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
            loss += loss_pos.sum() + loss_neg.sum()

        loss /= batch_size
        return loss

    def local_align_loss(self, part_embed, attribute_embed, labels, part_masks, attr_masks):    # [5, 64, 256], [5, 64, 256], [64,], [64, 5], [64, 5]
        alpha = 0.6
        beta = 0.4
        scale_pos = 10
        scale_neg = 40
        topK = 8

        batch_size = labels.size(0)     # 64
        part_embed = F.normalize(part_embed, p=2, dim=2)                # [5, 64, 256]
        attribute_embed = F.normalize(attribute_embed, p=2, dim=2)      # [5, 64, 256]
        labels_ = labels.expand(batch_size, batch_size).eq(
            labels.expand(batch_size, batch_size).t())                  # [64, 64], True, False

        losses = 0
        for i in range(self.num_parts):         # 5
            part_mask = part_masks[:, i]        # [64,]
            attr_mask = attr_masks[:, i]        # [64,]
            similarity = torch.matmul(part_embed[i], attribute_embed[i].t())        # [64, 64]
            rank1 = torch.argsort(similarity, dim=1, descending=True)               # [64, 64]
            rank2 = torch.argsort(similarity.t(), dim=1, descending=True)           # [64, 64]

            loss = 0
            for j in range(batch_size):
                if part_mask[j] == 0:
                    continue
                pred = similarity[j, attr_mask]         # [55,]
                # k-reciprocal sample
                label = labels_[j, :].float()           # [64,]
                forward_k_idx = rank1[j, :topK]         # [8,]   topK=8
                backward_k_idx = rank2[forward_k_idx, :topK]                    # [8, 8]
                sample_pos_idx = torch.nonzero(backward_k_idx == j, as_tuple=False)[:, 0]
                sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
                label[sample_pos_idx] = 1               # [64,]
                label = label[attr_mask]                # [55,]
                pos_inds = torch.nonzero(label == 1, as_tuple=False).squeeze(1)         # [3,]
                neg_inds = torch.nonzero(label == 0, as_tuple=False).squeeze(1)         # [54,]
                if pos_inds.numel() > 0:
                    loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
                    loss += loss_pos.sum()              # 16.7916
                if neg_inds.numel() > 0:
                    loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                    loss += loss_neg.sum()              # 16.7917

                if attr_mask[j] == 0:
                    continue
                pred = similarity[part_mask, j]         # [58,]
                # k-reciprocal sample
                label = labels_[j, :].float()           # [64,]
                forward_k_idx = rank2[j, :topK]
                backward_k_idx = rank1[forward_k_idx, :topK]
                sample_pos_idx = torch.nonzero(backward_k_idx == j, as_tuple=False)[:, 0]
                sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
                label[sample_pos_idx] = 1
                label = label[part_mask]
                pos_inds = torch.nonzero(label == 1, as_tuple=False).squeeze(1)
                neg_inds = torch.nonzero(label == 0, as_tuple=False).squeeze(1)
                if pos_inds.numel() > 0:
                    loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
                    loss += loss_pos.sum()
                if neg_inds.numel() > 0:
                    loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                    loss += loss_neg.sum()

            loss /= batch_size          # 39.5266
            losses += loss
        losses /= self.num_parts        # 31.6072
        return losses

    def forward(self, visual_embed, textual_embed=None, labels=None):
        if textual_embed is not None:
            global_align_loss = self.global_align_loss(visual_embed, textual_embed, labels)     # 46.7397
            # local_align_loss = self.local_align_loss(part_embed, attribute_embed, labels, vmask, tmask)       # 31.6072
            instance_loss, loss_v, loss_t = self.instance_loss(visual_embed, textual_embed, labels)             # 22.3195

            losses = {
                "instance_loss": instance_loss,
                "global_align_loss": global_align_loss,
                "loss_v": loss_v,
                "loss_t": loss_t,
                # "local_align_loss": local_align_loss
            }
        else:
            loss_v = self.image_loss(visual_embed, labels)  # 22.3195
            losses = {
                "instance_loss": loss_v,
                "loss_v": loss_v,
            }

        return losses


def make_loss_evaluator(cfg):
    return LossComputation(cfg)

import torch
import numpy as np
import cv2
import math
import os
import time
import sys
import shutil
from copy import deepcopy #Added
USE_TENSORBOARD = True
try:
    import tensorboardX
    #print('Using tensorboardX')
except:
    USE_TENSORBOARD = False

########### UTILS ####################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])

def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2, tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)

#/src/lib/models/decode.py decoding
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _left_aggregate(heat):
    '''heat: batchsize x channels x h x w'''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)

def _right_aggregate(heat):
    '''heat: batchsize x channels x h x w'''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i +1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)

def _top_aggregate(heat):
    '''heat: batchsize x channels x h x w'''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _bottom_aggregate(heat):
    '''heat: batchsize x channels x h x w'''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + aggr_weight * _right_aggregate(heat) + heat

def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + aggr_weight * _bottom_aggregate(heat) + heat

def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def agnex_ct_decode(t_heat, l_heat, b_heat, r_heat, ct_heat, t_regr=None, l_regr=None, b_regr=None, r_regr=None, K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000):

    batch, cat, height, width = t_heat.size()

    if aggr_weight > 0:
        t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
        l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
        b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
        r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)

    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)

    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, _, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, _, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, _, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, _, r_ys, r_xs = _topk(r_heat, K=K)

    ct_heat_agn, ct_clses = torch.max(ct_heat, dim=1, keepdim=True)

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()

    ct_inds     = box_ct_ys * width + box_ct_xs
    ct_inds     = ct_inds.view(batch, -1)
    ct_heat_agn = ct_heat_agn.view(batch, -1, 1)
    ct_clses    = ct_clses.view(batch, -1, 1)
    ct_scores   = _gather_feat(ct_heat_agn, ct_inds)
    clses       = _gather_feat(ct_clses, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + (b_scores < scores_thresh) + (r_scores < scores_thresh) + (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5

    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()

    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, b_xs, b_ys, r_xs, r_ys, clses], dim=2)
    return detections

def exct_decode(t_heat, l_heat, b_heat, r_heat, ct_heat, t_regr=None, l_regr=None, b_regr=None, r_regr=None, K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000):
    batch, cat, height, width = t_heat.size()
    if aggr_weight > 0:
        t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
        l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
        b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
        r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)

    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)

    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, t_clses, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, l_clses, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, b_clses, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, r_clses, r_ys, r_xs = _topk(r_heat, K=K)

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    t_clses = t_clses.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_clses = l_clses.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_clses = b_clses.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_clses = r_clses.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()
    ct_inds = t_clses.long() * (height * width) + box_ct_ys * width + box_ct_xs
    ct_inds = ct_inds.view(batch, -1)
    ct_heat = ct_heat.view(batch, -1, 1)
    ct_scores = _gather_feat(ct_heat, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    cls_inds = (t_clses != l_clses) + (t_clses != b_clses) + (t_clses != r_clses)
    cls_inds = (cls_inds > 0)

    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
              (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
              (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - cls_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5

    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = t_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()

    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, b_xs, b_ys, r_xs, r_ys, clses], dim=2)
    return detections

def ddd_decode(heat, rot, depth, dim, wh=None, reg=None, K=40):
    batch, cat, height, width = heat.size()
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)

    if wh is not None:
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        detections = torch.cat([xs, ys, scores, rot, depth, dim, wh, clses], dim=2)
    else:
        detections = torch.cat([xs, ys, scores, rot, depth, dim, clses], dim=2)
    return detections

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections

def multi_pose_decode(heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(0, 2, 1, 3).contiguous() # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3) # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    return detections

def gen_position(kps,dim,rot,meta,const):
    b=kps.size(0)
    c=kps.size(1)
    opinv=meta['trans_output_inv']
    calib=meta['calib']
    opinv = opinv.unsqueeze(1)
    opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()
    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
    hom = torch.ones(b, c, 1, 9).cuda()
    kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)
    kps = torch.bmm(opinv, kps).view(b, c, 2, 9)
    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
    si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]
    alpha_idx = alpha_idx.float()
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
    alpna_pre = alpna_pre.unsqueeze(2)
    rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi
    calib = calib.unsqueeze(1)
    calib = calib.expand(b, c, -1, -1).contiguous()
    kpoint = kps[:, :, :16]
    f = calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(kpoint)
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2)
    cxy = cxy.repeat(1, 1, 8)  # b,c,16
    kp_norm = (kpoint - cxy) / f
    l = dim[:, :, 2:3]
    h = dim[:, :, 0:1]
    w = dim[:, :, 1:2]
    cosori = torch.cos(rot_y)
    sinori = torch.sin(rot_y)
    B = torch.zeros_like(kpoint)
    C = torch.zeros_like(kpoint)
    kp = kp_norm.unsqueeze(3)  # b,c,16,1
    const = const.expand(b, c, -1, -1)
    A = torch.cat([const, kp], dim=3)

    B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 1:2] = h * 0.5
    B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 3:4] = h * 0.5
    B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 5:6] = h * 0.5
    B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 7:8] = h * 0.5
    B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 9:10] = -h * 0.5
    B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 11:12] = -h * 0.5
    B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 13:14] = -h * 0.5
    B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 15:16] = -h * 0.5

    C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori

    B = B - kp_norm * C
    AT = A.permute(0, 1, 3, 2)
    AT = AT.view(b * c, 3, 16)
    A = A.view(b * c, 16, 3)
    B = B.view(b * c, 16, 1).float()
    pinv = torch.bmm(AT, A)
    pinv = torch.inverse(pinv)  # b*c 3 3
    pinv = torch.bmm(pinv, AT)
    pinv = torch.bmm(pinv, B)
    pinv = pinv.view(b, c, 3, 1).squeeze(3)
    return pinv,rot_y,kps

def car_pose_decode(heat, wh, kps,dim,rot, prob=None,reg=None, hm_hp=None, hp_offset=None, K=100,meta=None,const=None):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2

    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    prob = _transpose_and_gather_feat(prob, inds)[:,:,0]
    prob = prob.view(batch, K, 1)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5
        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, K, num_joints * 2)
        hm_score=hm_score.permute(0, 2, 1, 3).squeeze(3).contiguous()
    position,rot_y,kps_inv=gen_position(kps,dim,rot,meta,const)
    detections = torch.cat([bboxes, scores, kps_inv,dim,hm_score,rot_y,position,prob,clses], dim=2)
    return detections

def car_pose_decode_faster(heat, kps,dim,rot, prob,K=100,meta=None,const=None):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2

    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    clses = clses.view(batch, K, 1).float()
    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    scores = scores.view(batch, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)

    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    prob = _transpose_and_gather_feat(prob, inds)[:,:,0]
    prob = prob.view(batch, K, 1)
    position,rot_y,kps_inv=gen_position(kps,dim,rot,meta,const)

    bboxes_kp=kps.view(kps.size(0),kps.size(1),9,2)
    box_min,_=torch.min(bboxes_kp,dim=2)
    box_max,_=torch.max(bboxes_kp,dim=2)
    bboxes=torch.cat((box_min,box_max),dim=2)
    hm_score=kps[:,:,0:9]
    detections = torch.cat([bboxes, scores, kps_inv,dim,hm_score,rot_y,position,prob,clses], dim=2)
    return detections

#/src/lib/utils/ddd_utils.py
def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2,0]
    y_corners = [0,0,0,0,-h,-h,-h,-h,-h/2]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2,0]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P,img_shape):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    h=img_shape[0]
    w=img_shape[1]
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d_center = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d_center = pts_2d_center[:, :2] / pts_2d_center[:, 2:]
    pts_2d=pts_2d_center[:9,:]
    pts_center=pts_2d_center[8:9,:]
    x_pts=pts_2d[:,0:1]
    y_pts=pts_2d[:,1:2]
    is_vis=np.ones(x_pts.shape)
    is_vis[x_pts>w]=0
    is_vis[y_pts>h]=0
    is_vis[x_pts < 0] = 0
    is_vis[y_pts < 0] = 0
    vis_num=is_vis.sum()
    is_vis=is_vis*2
    pts_2d=np.column_stack((pts_2d,is_vis))
    return pts_2d,vis_num,pts_center

def project_to_image3(pts_3d, P,img_shape):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    h=img_shape[0]
    w=img_shape[1]
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d_center = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d_center = pts_2d_center[:, :2] / pts_2d_center[:, 2:]
    pts_2d=pts_2d_center[:9,:]
    pts_center=pts_2d_center[8:9,:]
    x_pts=pts_2d[:,0:1]
    y_pts=pts_2d[:,1:2]
    is_vis=np.ones(x_pts.shape)
    is_vis[x_pts>w]=0
    is_vis[y_pts>h]=0
    is_vis[x_pts < 0] = 0
    is_vis[y_pts < 0] = 0
    vis_num=is_vis.sum()
    is_vis=is_vis*2
    f = P[0, 0]
    cx, cy = P[ 0, 2], P[1, 2]
    pts_2d[:,0]=(pts_2d[:,0]-cx)/f
    pts_2d[:, 1] = (pts_2d[:, 1] - cy)/ f
    pts_2d=np.column_stack((pts_2d,is_vis))
    return pts_2d,vis_num,pts_center

def compute_orientation_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 2 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d = orientation_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(0, 0, 255)): #corners are changed to int(corners) to be able to draw
    face_idx = [[0,1,5,4], [1,2,6, 5], [2,3,7,6], [3,0,4,7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])), (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 1, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])), (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])), (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
    return image

def unproject_2d_to_3d(pt_2d, depth, P):
    # pts_2d: 2
    # depth: 1
    # P: 3 x 4
    # return: 3
    z = depth - P[2, 3]
    x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
    pt_3d = np.array([x, y, z], dtype=np.float32)
    return pt_3d

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha

def ddd2locrot(center, alpha, dim, depth, calib):
    # single image
    locations = unproject_2d_to_3d(center, depth, calib)
    locations[1] += dim[0] / 2
    rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
    return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
    box_3d = compute_box_3d(dim, location, rotation_y)
    box_2d = project_to_image(box_3d, calib)
    return box_2d

########### Image ####################################

def flip(img):
    return img[:, :, ::-1].copy()

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        idx = (masked_gaussian >= masked_heatmap).reshape(1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap

def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

#/src/lib/utils/kitti_read.py
def E2R(Ry, Rx, Rz):
    '''Combine Euler angles to the rotation matrix (right-hand)
        Inputs:
            Ry, Rx, Rz : rotation angles along  y, x, z axis only has Ry in the KITTI dataset
        Returns:
            3 x 3 rotation matrix
    '''
    R_yaw = np.array([[math.cos(Ry), 0, math.sin(Ry)], [0, 1, 0], [-math.sin(Ry), 0, math.cos(Ry)]])
    R_pitch = np.array([[1, 0, 0], [0, math.cos(Rx), -math.sin(Rx)], [0, math.sin(Rx), math.cos(Rx)]])
    # R_roll = np.array([[[m.cos(Rz), -m.sin(Rz), 0], [m.sin(Rz), m.cos(Rz), 0], [ 0,         0 ,     1]])
    return (R_pitch.dot(R_yaw))

#/src/lib/utils/vis_3d_utils.py
def Space2Bev(P0, side_range=(-20, 20), fwd_range=(0, 70), res=0.1):
    x_img = (P0[0] / res).astype(np.int32)
    y_img = (-P0[2] / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.floor(fwd_range[1] / res)) - 1

    return np.array([x_img, y_img])

def vis_create_bev(width=750, side_range=(-20, 20), fwd_range=(0, 70), min_height=-2.5, max_height=1.5):
    ''' Project pointcloud to bev image for simply visualization
        Inputs: pointcloud:     3 x N in camera 2 frame
        Return: cv color image
    '''
    res = float(fwd_range[1] - fwd_range[0]) / width
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[:, :] = 255
    im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im_rgb

def vis_box_in_bev(im_bev, pos, dims, orien, width=750, gt=False,score=None, side_range=(-20, 20), fwd_range=(0, 70), min_height=-2.73, max_height=1.27):
    ''' Project 3D bounding box to bev image for simply visualization. It should use consistent width and side/fwd range input with the function: vis_lidar_in_bev
        Inputs: im_bev:         cv image
            pos, dim, orien: params of the 3D bounding box
        Return: cv color image
    '''
    dim = dims.copy()
    buf = dim.copy()
    res = float(fwd_range[1] - fwd_range[0]) / width

    R = E2R(orien, 0, 0)
    pts3_c_o = []
    pts2_c_o = []

    pts3_c_o.append(pos + R.dot(np.array([dim[0] / 2., 0, dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([dim[0] / 2, 0, -dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-dim[0] / 2, 0, -dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-dim[0] / 2, 0, dim[2] / 2.0]).T))

    pts3_c_o.append(pos + R.dot([dim[0] / 1.5, 0, 0]))
    pts2_bev = []
    for index in range(5):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range, fwd_range=fwd_range, res=res))

    if gt is False:
        lineColor3d = (100, 100, 0)
    else:
        lineColor3d = (0, 0, 255)
    if gt == 'next':
        lineColor3d = (255, 0, 0)
    if gt == 'g':
        lineColor3d = (0, 255, 0)
    if gt == 'b':
        lineColor3d = (255, 0, 0)
    if gt == 'n':
        lineColor3d = (5, 100, 100)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)

    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)

    if score is not None:
        show_text(im_bev,pts2_bev[4],score)
    return im_bev

def show_text(img, cor, score):
    txt = '{:.2f}'.format(score)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, txt, (cor[0], cor[1]), font, 0.3, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img

########### post_process ####################################

def car_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = dets[i, :, 5:23]
    dim = dets[i, :, 23:26]
    rot_y=dets[i,:,35:36]
    pts_score=dets[i, :, 26:35]
    prob=dets[i, :, 39:40]
    position=dets[i,:,36:39]
    cat = dets[i, :, 40:41]
    top_preds = np.concatenate([bbox.reshape(-1, 4), dets[i, :, 4:5], pts,pts_score,dim,rot_y,position,prob,cat], axis=1).astype(np.float32).tolist()
    #bbox score kps kps_score dim rot_y position prob
    #0:4  4:5   5:23 23:32    32:35 35:36 36:39 39:40
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

########### external/nms ####################################
#def soft_nms_39(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):
def soft_nms_39(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    #Implementation uses python variables instead of cython to avoid using C. Original cython definitions are commented against each variable below
    N = int(boxes.shape[0]) #cdef unsigned int
    iw, ih, box_area = float(0.0), float(0.0), float(0.0) #cdef float
    ua = float(0.0) #cdef float
    pos = int(0) #cdef int
    maxscore = float(0.0) #cdef float
    maxpos = int(0) #cdef int
    x1,x2,y1,y2 = float(0.0), float(0.0), float(0.0), float(0.0) #cdef float
    tx1,tx2,ty1,ty2 = float(0.0), float(0.0), float(0.0), float(0.0) #cdef float
    ts,area,weight,ov = float(0.0), float(0.0), float(0.0), float(0.0) #cdef float
    tmp = float(0.0) #cdef float

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i
        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        for j in range(5, 39):
            tmp = boxes[i, j]
            boxes[i, j] = boxes[maxpos, j]
            boxes[maxpos, j] = tmp

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        for j in range(5, 39):
                            tmp = boxes[pos, j]
                            boxes[pos, j] = boxes[N - 1, j]
                            boxes[N - 1, j] = tmp
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

########### KITTI ####################################

def boxes3d_to_corners3d_torch(boxes3d, flip=False):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return: corners_rotated: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l, ry = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
    if flip:
        ry = ry + np.pi
    centers = boxes3d[:, 0:3]
    zeros = torch.cuda.FloatTensor(boxes_num, 1).fill_(0)
    ones = torch.cuda.FloatTensor(boxes_num, 1).fill_(1)

    x_corners = torch.cat([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dim=1)  # (N, 8)
    y_corners = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=1)  # (N, 8)
    z_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim=1)  # (N, 8)
    corners = torch.cat((x_corners.unsqueeze(dim=1), y_corners.unsqueeze(dim=1), z_corners.unsqueeze(dim=1)), dim=1) # (N, 3, 8)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, zeros, sina], dim=1)
    raw_2 = torch.cat([zeros, ones, zeros], dim=1)
    raw_3 = torch.cat([-sina, zeros, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1), raw_3.unsqueeze(dim=1)), dim=1)  # (N, 3, 3)

    corners_rotated = torch.matmul(R, corners)  # (N, 3, 8)
    corners_rotated = corners_rotated + centers.unsqueeze(dim=2).expand(-1, -1, 8)
    corners_rotated = corners_rotated.permute(0, 2, 1)
    return corners_rotated

def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev

########### logger.py ####################################
class Logger(object):
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        if not os.path.exists(opt.debug_dir):
            os.makedirs(opt.debug_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
        file_name = opt.save_dir + 'opt.txt' #os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = opt.save_dir + '/logs_{}'.format(time_str)
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(log_dir):#(os.path.dirname(log_dir)):
                #os.mkdir(os.path.dirname(log_dir))
                os.mkdirs(log_dir)
            #if not os.path.exists(log_dir):
                #os.mkdir(log_dir)
        self.log = open(log_dir + '/log.txt', 'w')
        try:
            #os.system('cp {}/opt.txt {}/'.format(opt.save_dir, log_dir))
            shutil.copy(opt.save_dir+'opt.txt', log_dir)
        except:
            pass
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)


########### Debugger.py ####################################
class Debugger(object):

    def __init__(self, ipynb=False, theme='black', num_classes=-1, dataset=None, down_ratio=4, det_cats=[]):
        self.ipynb = ipynb
        if self.ipynb:
            import matplotlib.pyplot as plt
            self.plt = plt
        self.imgs = {}
        self.results = {} #Added
        self.entry_results = result_dict = {'class':[], 'alpha':[], #Added
                'bbox0':[], 'bbox1':[], 'bbox2':[], 'bbox3':[],
                'dim0':[], 'dim1':[], 'dim2':[],
                'posX':[], 'posY':[], 'posZ':[],
                'ori':[], 'score':[],
                'locker0':[], 'locker1':[], 'locker2':[]}
        self.BEV={}
        self.theme = theme
        colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
        self.dim_scale = 1
        if dataset == 'coco_hp':
            self.names = ['p']
            self.num_class = 1
            self.num_joints = 17
            self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
            self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 255),(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                        (255, 0, 255), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
            self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),
                                (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255)]
        elif dataset == 'kitti_hp':
            self.names = kitti_class_name
            self.num_class = 1
            self.num_joints = 9
            self.edges1 = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7]]
            self.edges = [[0, 1],[1,2],[2,3],[3,0],[4, 5], [5, 6], [6, 7],[0, 4],[1,5],[2,6],[3,7]]
            self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0), (255, 0, 0)]
            self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),(255, 100, 50)]
        elif num_classes == 80 or dataset == 'coco':
            self.names = coco_class_name
        elif num_classes == 20 or dataset == 'pascal':
            self.names = pascal_class_name
        elif dataset == 'gta':
            self.names = gta_class_name
            self.focal_length = 935.3074360871937
            self.W = 1920
            self.H = 1080
            self.dim_scale = 3
        elif dataset == 'viper':
            self.names = gta_class_name
            self.focal_length = 1158
            self.W = 1920
            self.H = 1080
            self.dim_scale = 3
        elif num_classes == 3 or dataset == 'kitti':
            self.names = kitti_class_name
            self.focal_length = 721.5377
            self.W = 1242
            self.H = 375
        elif num_classes == 3 or dataset == 'kitti_hp':
            self.names = kitti_class_name
            self.focal_length = 721.5377
            self.W = 1242
            self.H = 375
        elif dataset == 'custom':
            self.names = det_cats
            self.num_joints = 9
            self.edges = [[0, 1],[1,2],[2,3],[3,0],[4, 5], [5, 6], [6, 7],[0, 4],[1,5],[2,6],[3,7]]
            self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0), (255, 0, 0)]
            self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),(255, 100, 50)]
        num_classes = len(self.names)
        self.down_ratio=down_ratio
        # for bird view
        self.world_size = 64
        self.out_size = 384

    def add_img(self, img, img_id='default', revert_color=False):
        if revert_color:
            img = 255 - img
        self.imgs[img_id] = img.copy()
        im_bev = vis_create_bev(width=img.shape[0] * 2)
        self.BEV[img_id] = im_bev
        self.results[img_id] = deepcopy(self.entry_results)

    def add_bev(self, box,img_id, color=(255,0,0), is_faster = True):
        position=box[36:39]
        dim=box[32:35]
        l = dim[2]
        h = dim[0]
        w = dim[1]
        ori=box[35]
        if is_faster:
            score_3d = box[4] * (1 / (1 + math.exp(-box[39])))
        else:
            score_3d = (box[4] + (1 / (1 + math.exp(-box[39]))) + (sum(box[23:32]) / 9)) / 3
        self.BEV[img_id] = vis_box_in_bev(self.BEV[img_id], position, [l,h,w], ori, score=score_3d, width=self.imgs[img_id].shape[0] * 2, gt=color)

    def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
        self.imgs[imgId] = (mask.reshape(mask.shape[0], mask.shape[1], 1) * 255 * trans + bg * (1 - trans)).astype(np.uint8)

    def show_img(self, pause = False, imgId = 'default'):
        cv2.imshow('{}'.format(imgId), self.imgs[imgId])
        if pause:
            cv2.waitKey()

    def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
        if self.theme == 'white':
            fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    def gen_colormap(self, img, output_res=None):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    def gen_colormap_hp(self, img, output_res=None):
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    def add_rect(self, rect1, rect2, c, conf=1, img_id='default'):
        cv2.rectangle(self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
        if conf < 1:
            cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

    def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'):
        bbox = np.array(bbox, dtype=np.int32)
        cat = int(cat)
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        txt = '{}{:.1f}'.format(self.names[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        if show_txt:
            cv2.rectangle(self.imgs[img_id], (bbox[0], bbox[1] - cat_size[1] - 2), (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def add_coco_hp(self, points, img_id='default'):
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
        for j in range(self.num_joints):
            cv2.circle(self.imgs[img_id], (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
        for j, e in enumerate(self.edges):
            if points[e].min() > 0:
                cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]), (points[e[1], 0], points[e[1], 1]), self.ec[j], 2, lineType=cv2.LINE_AA)

    def add_kitti_hp(self, points, img_id='default'):
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
        for j in range(self.num_joints):
            cv2.circle(self.imgs[img_id], (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)

    def save_kitti_hp_point(self, points_dim, img_path,opt,img_id='default'):
        result_dir=opt.exp_dir
        file_number=img_path.split('.')[-2][-6:]
        self.write_points_results(result_dir,file_number,points_dim)

    def save_kitti_format(self, results, opt, locker, img_path='', img_id='default',is_faster=False):
        if img_path !='':
            result_dir = ''
            file_number = img_id
        else:
            result_dir=opt.results_dir
            file_number=img_path.split('.')[-2][-6:]
        box=results[:4]
        if is_faster:
            score=results[4]*(1/(1+math.exp(-results[39])))
        else:
            score=(results[4]+(1/(1+math.exp(-results[39])))+(sum(results[23:32])/9))/3
        dim=results[32:35]
        if dim[0] < 0 or dim[1]<0 or dim[2]<0:
            print('WARN: at least one dimension is less that zero')#file_number,dim)
        pos=results[36:39]
        ori=results[35]
        cat=results[40]
        det_cats = opt.det_cats#['Car', 'Pedestrian', 'Cyclist']
        self.write_detection_results(det_cats[int(cat)],result_dir,file_number,box,dim,pos,ori,score, locker)

    def write_detection_results(self,cls, result_dir, file_number, box,dim,pos,ori,score,locker):
        '''One by one write detection results to KITTI format label files.'''
        Px = pos[0]
        Py = pos[1]
        Pz = pos[2]
        l =dim[2]
        h = dim[0]
        w = dim[1]
        Py=Py+h/2
        pi=np.pi
        if ori > 2 * pi:
            while ori > 2 * pi:
                ori -= 2 * pi
        if ori < -2 * pi:
            while ori < -2 * pi:
                ori += 2 * pi

        if ori > pi:
            ori = 2 * pi - ori
        if ori < -pi:
            ori = 2 * pi + pi

        alpha = ori - math.atan2(Px, Pz)
        # convert the object from cam2 to the cam0 frame

        #Create Result Dict
        self.results[file_number]['class'].append(cls)
        self.results[file_number]['alpha'].append(alpha)
        self.results[file_number]['bbox0'].append(box[0])
        self.results[file_number]['bbox1'].append(box[1])
        self.results[file_number]['bbox2'].append(box[2])
        self.results[file_number]['bbox3'].append(box[3])
        self.results[file_number]['dim0'].append(h)
        self.results[file_number]['dim1'].append(w)
        self.results[file_number]['dim2'].append(l)
        self.results[file_number]['posX'].append(Px)
        self.results[file_number]['posY'].append(Py)
        self.results[file_number]['posZ'].append(Pz)
        self.results[file_number]['ori'].append(ori)
        self.results[file_number]['score'].append(score)
        self.results[file_number]['locker0'].append(locker[0])
        self.results[file_number]['locker1'].append(locker[1])
        self.results[file_number]['locker2'].append(locker[2])

        #result_dict['class'] = result_dict['class'].append(cls)
        #result_dict['alpha'] = result_dict['alpha'].append(alpha)
        #result_dict['bbox0'] = result_dict['bbox0'].append(box[0])
        #result_dict['bbox1'] = result_dict['bbox1'].append(box[1])
        #result_dict['bbox2'] = result_dict['bbox2'].append(box[2])
        #result_dict['bbox3'] = result_dict['bbox3'].append(box[3])
        #result_dict['dim0'] = result_dict['dim0'].append(h)
        #result_dict['dim1'] = result_dict['dim1'].append(w)
        #result_dict['dim2'] = result_dict['dim2'].append(l)
        #result_dict['posX'] = result_dict['posX'].append(Px)
        #result_dict['posY'] = result_dict['posY'].append(Py)
        #result_dict['posZ'] = result_dict['posZ'].append(Pz)
        #result_dict['ori'] = result_dict['ori'].append(ori)
        #result_dict['score'] = result_dict['score'].append(score)
        #result_dict['locker0'] = result_dict['locker0'].append(locker[0])
        #result_dict['locker1'] = result_dict['locker1'].append(locker[1])
        #result_dict['locker2'] = result_dict['locker2'].append(locker[2])

        if result_dir !='': #Restructured to handle both file creation and dict return
            result_dir = result_dir + '/data'
            output_str = cls + ' '
            #output_str += '%.2f %.d ' % (-1, -1)
            output_str += '%.7f %.7f %.7f %.7f %.7f ' % (alpha, box[0], box[1], box[2], box[3])
            output_str += '%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f \n' % (h, w, l, Px, Py, Pz, ori, score, locker[0], locker[1], locker[2])

            # Write TXT files
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            pred_filename = result_dir + '/' + file_number + '.txt'
            if not os.path.isfile(pred_filename):
                with open(pred_filename,'w') as f:
                    f.write('Class alpha bbox[0] bbox[1] bbox[2] bbox[3] h w l Px Py Pz ori score locker_1 locker_2 locker_3\n')
            with open(pred_filename, 'a') as det_file:
                det_file.write(output_str)
            return result_dict

    def write_points_results(self, result_dir, file_number,point_dim):
        '''One by one write detection results to KITTI format label files.'''
        if result_dir is None:
            return
        result_dir = result_dir + '/sslrtm3d/56'

        # convert the object from cam2 to the cam0 frame
        output_str=' '
        for w in range(len(point_dim)):
            output_str += '%.2f ' % (point_dim[w])
        output_str += '\n'

        # Write TXT files
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        pred_filename = result_dir + '/' + file_number + '.txt'
        with open(pred_filename, 'a') as det_file:
            det_file.write(output_str)

    def add_points(self, points, img_id='default'):
        num_classes = len(points)
        for i in range(num_classes):
            for j in range(len(points[i])):
                c = self.colors[i, 0, 0]
                cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, points[i][j][1] * self.down_ratio), 5, (255, 255, 255), -1)
                cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, points[i][j][1] * self.down_ratio), 3, (int(c[0]), int(c[1]), int(c[2])), -1)

    def show_all_imgs(self, pause=False, time=0):
        if not self.ipynb:
            for i, v in self.imgs.items():
                cv2.imshow('{}'.format(i), v)
            for i, v in self.BEV.items():
                cv2.imshow('{}BEV'.format(i), v)
            if cv2.waitKey(0 if pause else 1) == 27:
                import sys
                sys.exit(0)
        else:
            self.ax = None
            nImgs = len(self.imgs)
            fig=self.plt.figure(figsize=(nImgs * 10,10))
            nCols = nImgs
            nRows = nImgs // nCols
            for i, (k, v) in enumerate(self.imgs.items()):
                fig.add_subplot(1, nImgs, i + 1)
                if len(v.shape) == 3:
                    self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    self.plt.imshow(v)
            self.plt.show()

    def save_img(self, imgId='default', path='./cache/debug/'):
        cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])

    def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
        if genID:
            try:
                idx = int(np.loadtxt(path + '/id.txt'))
            except:
                idx = 0
            prefix=idx
            np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

    def remove_side(self, img_id, img):
        if not (img_id in self.imgs):
            return
        ws = img.sum(axis=2).sum(axis=0)
        l = 0
        while ws[l] == 0 and l < len(ws):
            l+= 1
        r = ws.shape[0] - 1
        while ws[r] == 0 and r > 0:
            r -= 1
        hs = img.sum(axis=2).sum(axis=1)
        t = 0
        while hs[t] == 0 and t < len(hs):
            t += 1
        b = hs.shape[0] - 1
        while hs[b] == 0 and b > 0:
            b -= 1
        self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

    def project_3d_to_bird(self, pt):
        pt[0] += self.world_size / 2
        pt[1] = self.world_size - pt[1]
        pt = pt * self.out_size / self.world_size
        return pt.astype(np.int32)

    def add_ct_detection(self, img, dets, show_box=False, show_txt=True,  center_thresh=0.5, img_id='det'):
        # dets: max_preds x 5
        self.imgs[img_id] = img.copy()
        if type(dets) == type({}):
            for cat in dets:
                for i in range(len(dets[cat])):
                    if dets[cat][i, 2] > center_thresh:
                        cl = (self.colors[cat, 0, 0]).tolist()
                        ct = dets[cat][i, :2].astype(np.int32)
                        if show_box:
                            w, h = dets[cat][i, -2], dets[cat][i, -1]
                            x, y = dets[cat][i, 0], dets[cat][i, 1]
                            bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)
                            self.add_coco_bbox(bbox, cat - 1, dets[cat][i, 2], show_txt=show_txt, img_id=img_id)
        else:
            for i in range(len(dets)):
                if dets[i, 2] > center_thresh:
                    cat = int(dets[i, -1])
                    cl = (self.colors[cat, 0, 0] if self.theme == 'black' else 255 - self.colors[cat, 0, 0]).tolist()
                    ct = dets[i, :2].astype(np.int32) * self.down_ratio
                    cv2.circle(self.imgs[img_id], (ct[0], ct[1]), 3, cl, -1)
                    if show_box:
                        w, h = dets[i, -3] * self.down_ratio, dets[i, -2] * self.down_ratio
                        x, y = dets[i, 0] * self.down_ratio, dets[i, 1] * self.down_ratio
                        bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)
                        self.add_coco_bbox(bbox, dets[i, -1], dets[i, 2], img_id=img_id)

    def add_3d_detection(self, results, calib, img_id='default',  locker=[0,0,0], show_txt=False):
        dim = results[32:35]
        pos = results[36:39]
        ori = results[35]
        cat = int(results[40])
        pos[1] = pos[1] + dim[0] / 2
        cl = (self.colors[cat, 0, 0]).tolist()
        box_3d = compute_box_3d(dim, pos, ori)
        box_2d = self.project_to_image(box_3d, calib)
        self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
        locker_3d = compute_box_3d(locker, pos, ori)
        locker_2d = self.project_to_image(locker_3d, calib)
        self.imgs[img_id] = draw_box_3d(self.imgs[img_id], locker_2d, [0,255,0])
        return self.imgs[img_id] #ADDED

    def project_to_image(self, pts_3d, P):
        # pts_3d: n x 3
        # P: 3 x 4
        # return: n x 2
        pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
        pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d

    def compose_vis_add(self, img_path, dets, calib, center_thresh, pred, bev, img_id='out'):
        self.imgs[img_id] = cv2.imread(img_path)
        h, w = pred.shape[:2]
        hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
        self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
        self.add_blend_img(self.imgs[img_id], pred, img_id)
        for cat in dets:
            for i in range(len(dets[cat])):
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                if dets[cat][i, -1] > center_thresh:
                    dim = dets[cat][i, 5:8]
                    loc  = dets[cat][i, 8:11]
                    rot_y = dets[cat][i, 11]
                    if loc[2] > 1:
                        box_3d = compute_box_3d(dim, loc, rot_y)
                        box_2d = project_to_image(box_3d, calib)
                        box_2d[:, 0] /= hs
                        box_2d[:, 1] /= ws
                        self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
        self.imgs[img_id] = np.concatenate([self.imgs[img_id], self.imgs[bev]], axis=1)

    def add_2d_detection(self, img, dets, show_box=False, show_txt=True, center_thresh=0.5, img_id='det'):
        self.imgs[img_id] = img
        for cat in dets:
            for i in range(len(dets[cat])):
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                if dets[cat][i, -1] > center_thresh:
                    bbox = dets[cat][i, 1:5]
                    self.add_coco_bbox(bbox, cat - 1, dets[cat][i, -1], show_txt=show_txt, img_id=img_id)

    def add_bird_view(self, dets, center_thresh=0.3, img_id='bird'):
        bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
        for cat in dets:
            cl = (self.colors[cat - 1, 0, 0]).tolist()
            lc = (250, 152, 12)
            for i in range(len(dets[cat])):
                if dets[cat][i, -1] > center_thresh:
                    dim = dets[cat][i, 5:8]
                    loc  = dets[cat][i, 8:11]
                    rot_y = dets[cat][i, 11]
                    rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
                    for k in range(4):
                        rect[k] = self.project_3d_to_bird(rect[k])
                    cv2.polylines(bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)], True,lc,2,lineType=cv2.LINE_AA)
                    for e in [[0, 1]]:
                        t = 4 if e == [0, 1] else 1
                        cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]), (rect[e[1]][0], rect[e[1]][1]), lc, t, lineType=cv2.LINE_AA)
        self.imgs[img_id] = bird_view

    def add_bird_views(self, dets_dt, dets_gt, center_thresh=0.3, img_id='bird'):
        alpha = 0.5
        bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
        for ii, (dets, lc, cc) in enumerate([(dets_gt, (12, 49, 250), (0, 0, 255)), (dets_dt, (250, 152, 12), (255, 0, 0))]):
            for cat in dets:
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                for i in range(len(dets[cat])):
                    if dets[cat][i, -1] > center_thresh:
                        dim = dets[cat][i, 5:8]
                        loc  = dets[cat][i, 8:11]
                        rot_y = dets[cat][i, 11]
                        rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
                        for k in range(4):
                            rect[k] = self.project_3d_to_bird(rect[k])
                        if ii == 0:
                            cv2.fillPoly(bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)], lc,lineType=cv2.LINE_AA)
                        else:
                            cv2.polylines(bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)], True,lc,2,lineType=cv2.LINE_AA)
                        for e in [[0, 1]]:
                            t = 4 if e == [0, 1] else 1
                            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]), (rect[e[1]][0], rect[e[1]][1]), lc, t, lineType=cv2.LINE_AA)
        self.imgs[img_id] = bird_view

kitti_class_name = ['Car', 'Pedestrian', 'Cyclist']
gta_class_name = ['p', 'v']
pascal_class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"]
coco_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                            'scissors', 'teddy bear', 'hair drier', 'toothbrush']

color_list = np.array(
                            [
                            1.000, 1.000, 1.000,
                            0.850, 0.325, 0.098,
                            0.929, 0.694, 0.125,
                            0.494, 0.184, 0.556,
                            0.466, 0.674, 0.188,
                            0.301, 0.745, 0.933,
                            0.635, 0.078, 0.184,
                            0.300, 0.300, 0.300,
                            0.600, 0.600, 0.600,
                            1.000, 0.000, 0.000,
                            1.000, 0.500, 0.000,
                            0.749, 0.749, 0.000,
                            0.000, 1.000, 0.000,
                            0.000, 0.000, 1.000,
                            0.667, 0.000, 1.000,
                            0.333, 0.333, 0.000,
                            0.333, 0.667, 0.000,
                            0.333, 1.000, 0.000,
                            0.667, 0.333, 0.000,
                            0.667, 0.667, 0.000,
                            0.667, 1.000, 0.000,
                            1.000, 0.333, 0.000,
                            1.000, 0.667, 0.000,
                            1.000, 1.000, 0.000,
                            0.000, 0.333, 0.500,
                            0.000, 0.667, 0.500,
                            0.000, 1.000, 0.500,
                            0.333, 0.000, 0.500,
                            0.333, 0.333, 0.500,
                            0.333, 0.667, 0.500,
                            0.333, 1.000, 0.500,
                            0.667, 0.000, 0.500,
                            0.667, 0.333, 0.500,
                            0.667, 0.667, 0.500,
                            0.667, 1.000, 0.500,
                            1.000, 0.000, 0.500,
                            1.000, 0.333, 0.500,
                            1.000, 0.667, 0.500,
                            1.000, 1.000, 0.500,
                            0.000, 0.333, 1.000,
                            0.000, 0.667, 1.000,
                            0.000, 1.000, 1.000,
                            0.333, 0.000, 1.000,
                            0.333, 0.333, 1.000,
                            0.333, 0.667, 1.000,
                            0.333, 1.000, 1.000,
                            0.667, 0.000, 1.000,
                            0.667, 0.333, 1.000,
                            0.667, 0.667, 1.000,
                            0.667, 1.000, 1.000,
                            1.000, 0.000, 1.000,
                            1.000, 0.333, 1.000,
                            1.000, 0.667, 1.000,
                            0.167, 0.000, 0.000,
                            0.333, 0.000, 0.000,
                            0.500, 0.000, 0.000,
                            0.667, 0.000, 0.000,
                            0.833, 0.000, 0.000,
                            1.000, 0.000, 0.000,
                            0.000, 0.167, 0.000,
                            0.000, 0.333, 0.000,
                            0.000, 0.500, 0.000,
                            0.000, 0.667, 0.000,
                            0.000, 0.833, 0.000,
                            0.000, 1.000, 0.000,
                            0.000, 0.000, 0.167,
                            0.000, 0.000, 0.333,
                            0.000, 0.000, 0.500,
                            0.000, 0.000, 0.667,
                            0.000, 0.000, 0.833,
                            0.000, 0.000, 1.000,
                            0.000, 0.000, 0.000,
                            0.143, 0.143, 0.143,
                            0.286, 0.286, 0.286,
                            0.429, 0.429, 0.429,
                            0.571, 0.571, 0.571,
                            0.714, 0.714, 0.714,
                            0.857, 0.857, 0.857,
                            0.000, 0.447, 0.741,
                            0.50, 0.5, 0]
                        ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

########### Select Locker ####################################
def sel_locker(opt,results):
    dim = results[32:35]
    locker_sizes= opt.locker_sizes
    locker_vols = [i[0]*i[1]*i[2] for i in locker_sizes]
    box_vol = dim[0]*dim[1]*dim[2]
    locker_vol = [[i,j,k] for [i,j,k] in locker_sizes if i*j*k>box_vol*1]
    locker_vol = sorted(locker_vol, key= lambda x: x[0]*x[1]*x[2] )
    idxs = [[0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0],[2,1,0]]
    sel_locker=[]
    found = False
    for l in locker_vol:
        for idx in idxs:
            tmp = [l[i] for i in idx]
            if tmp[0]>=dim[0] and tmp[1]>=dim[1] and tmp[2]>=dim[2]:
                sel_locker = tmp
                found = True
                break
        if found:
            break
    if len(sel_locker)==0:
        return [0,0,0]
    else:
        return sel_locker

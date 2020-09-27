import torch
import torch.nn as nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

from .vline_utils import compute_gt_bin_many
from .vline_utils import create_indgroupmaps
from .vline_utils import adjust_vps
from .vline_utils import adjust_bboxs
from .vline_utils import adjust_polygons
from .vline_utils import torch2numpy, find_min_max_angle, intersect_vp_bbox
from .vline_utils import compute_gt_bin_many_shapes_vert
from .vline_utils import compute_gt_bin_many_shapes_hor

import numpy as np
import cv2
import math
from bresenham import bresenham
from scipy import signal
import copy

def is_valid_box(box, THRED=5):
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    width = torch.abs(x2 - x1)
    height = torch.abs(y2 - y1)
    valid = np.logical_and(width.cpu().numpy() > THRED, height.cpu().numpy() > THRED)
    return valid

def shift_box(box, hor_shift, ver_shift):
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    x_shift = (x2 - x1) * hor_shift
    y_shift = (y2 - y1) * ver_shift
    box[:, 0] = box[:, 0] + x_shift
    box[:, 1] = box[:, 1] + y_shift
    box[:, 2] = box[:, 2] + x_shift
    box[:, 3] = box[:, 3] + y_shift

def add_margin_to_box(box, w_ratio = 1.25, h_ratio = 1.25, debug=None):
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    x_c = 0.5 * (x1 + x2)
    y_c = 0.5 * (y1 + y2)
    w_extend = 0.5 * (x2 - x1) * w_ratio
    h_extend = 0.5 * (y2 - y1) * h_ratio
    if debug:
        for ind, ratio_debug in debug.items():
            w_extend[ind] = 0.5 * (x2[ind] - x1[ind]) * ratio_debug
            h_extend[ind] = 0.5 * (y2[ind] - y1[ind]) * ratio_debug

    box[:, 0] = x_c - w_extend
    box[:, 1] = y_c - h_extend
    box[:, 2] = x_c + w_extend
    box[:, 3] = y_c + h_extend

def add_margin_to_box_with_margin(box, w_ratio = 1.25, h_ratio = 1.25, height = None, width = None):
    box_new = copy.deepcopy(box)
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    x_c = 0.5 * (x1 + x2)
    y_c = 0.5 * (y1 + y2)
    w_extend = 0.5 * (x2 - x1) * w_ratio
    h_extend = 0.5 * (y2 - y1) * h_ratio
    box_new[:, 0] = np.maximum(x_c - w_extend, 0)
    box_new[:, 1] = np.maximum(y_c - h_extend, 0)
    box_new[:, 2] = np.minimum(x_c + w_extend, width-1)
    box_new[:, 3] = np.minimum(y_c + h_extend, height-1)

    offsets = [box[:,0]-box_new[:,0], box[:,1]-box_new[:,1]]
    offsets = np.stack(offsets, axis=1)
    length_ori = [x2-x1, y2-y1]
    length_ori = np.stack(length_ori, axis=1)
    return box_new, offsets, length_ori

class VLineLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size, num_classes, bins, loss_w, is_focal=True, cls_agnostic_vline_reg=False, is_simple_bce=False, num_boundary=3):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.num_boundary = num_boundary
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.bins = bins
        self.is_simple_bce = is_simple_bce
        self.is_focal = is_focal
        self.loss_w = loss_w
        if self.is_focal:
            self.alpha = 0.25
            self.gamma = 2

            gaussian = signal.gaussian(3, std=1)
            self.nn_gaussian = nn.Conv1d(1, 1, 3, padding=1)
            self.nn_gaussian.weight.data.copy_(torch.from_numpy(gaussian))
            self.nn_gaussian.bias.data.fill_(0)

            self.nn_gaussian = self.nn_gaussian.cuda()
            self.beta = 4
        self.cls_agnostic_vline_reg = cls_agnostic_vline_reg


    def match_targets_to_proposals(self, proposal, target):
        # NOTE(H): Seems the code like to keep xyxy. Then let's keep xyxy all around! 
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        target = target.copy_with_fields(["labels", "vps", "vp_vert", "masks"])

        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets
        
    def __call__(self, proposals, vline_feats, gt_vline_inds, is_vert):
    # def __call__(self, proposals, vline_feats, gt_vline_inds, valid_vecs):
        """
        Arguments:
        Return:
        """
        
        if not self.cls_agnostic_vline_reg:
            device = vline_feats.device
            labels_pos = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            assert all(labels_pos > 0)
            map_inds = 3 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2], device=device)
            vline_feats_new = []
            for vline_feat, map_ind in zip(vline_feats, map_inds):
                vline_feats_new.append(vline_feat[:, map_ind][None, :, :])
            vline_feats = torch.cat(vline_feats_new, 0)

        vline_prob = self.sigmoid(vline_feats)
        if self.is_focal:
            if self.is_simple_bce:
                loss_classifier = self.loss_w * F.binary_cross_entropy_with_logits(vline_feats, gt_vline_inds.float(), size_average=True)
            else:             
                # Focal loss from CornerNet:
                # alpha in cornernet paper is the gamma in focal loss paper (=2)
                # y in cornernet replaces (but isn't) the alpha in focal loss
                # x = vline_feats * valid_vecs.float()
                x = vline_feats
                t = gt_vline_inds # NxLx2
                t1 = t[:,:,0][:,:,None].permute(0,2,1).float()
                t2 = t[:,:,1][:,:,None].permute(0,2,1).float()
                t3 = t[:,:,2][:,:,None].permute(0,2,1).float()
                t1 = self.nn_gaussian(t1).permute(0,2,1)
                t2 = self.nn_gaussian(t2).permute(0,2,1)
                t3 = self.nn_gaussian(t3).permute(0,2,1)
                tnew = torch.cat([t1, t2, t3],2)
                p = vline_prob
                pt = torch.where(t>0, 1-p, p)
                with torch.no_grad():
                    tnew = torch.where(t>0.9999, tnew, 1-tnew)
                    w = pt.pow(self.gamma)
                    beta = tnew.pow(self.beta)
                    w = beta*w
                loss_classifier = self.loss_w * F.binary_cross_entropy_with_logits(x, t.float(), w, size_average=True)

                # # Original focal loss
                # x = vline_feats * valid_vecs.float()
                # t = gt_vline_inds
                # p = vline_prob
                # pt = torch.where(t>0, p, 1-p)
                # with torch.no_grad():
                #     w = (1-pt).pow(self.gamma)
                #     w = torch.where(t>0, self.alpha*w, (1-self.alpha)*w)
                # loss_classifier = F.binary_cross_entropy_with_logits(x, t, w, size_average=True)
        else:
            if self.num_boundary == 1:
                loss_classifier = self.loss_ce(vline_feats[:,:,0], gt_vline_inds.long())
                # exit(0)
                return loss_classifier, None
            elif self.num_boundary == 3:
                # TODO(H): This is hard coded, in original vline gt is NxLx3
                # But in F-BP and Bi-BP, it's Nx1
                # x = vline_feats * valid_vecs.float()
                x = vline_feats
                # TODO(H): Need to correct the valid.
                losses = [] 
                for ind in range(self.num_boundary):
                    gt_vline_inds0 = gt_vline_inds[:,:,ind]
                    # Possible if no GT, max will be 0
                    valid_gt_vline_inds = torch.nonzero(torch.max(gt_vline_inds0, 1)[0] > 0.9)[:,0]
                    if valid_gt_vline_inds.size(0)!=0:
                        gts = torch.argmax(gt_vline_inds0[valid_gt_vline_inds], 1)
                        losses.append(self.loss_ce(x[:,:,ind][valid_gt_vline_inds], gts))
                # TOOD(H): Direct sum might be not real mean, but we can try it
                if len(losses) == 0:
                    loss_classifier = torch.zeros(1).cuda()
                else:
                    loss_classifier = losses[0]
                    for ind in range(1, len(losses)):
                        loss_classifier += losses[ind]

                w1 = 0.5
                w2 = 0.5
                w3 = 0.5
                preds = torch.argmax(vline_prob, 1) # Nx3
                LEAST_THRED = 5

                gt_vline_inds0 = gt_vline_inds[:,:,0]
                valid_gt_vline0 = torch.max(gt_vline_inds0, 1)[0] > 0.9
                gt_vline_inds1 = gt_vline_inds[:,:,1]
                valid_gt_vline1 = torch.max(gt_vline_inds1, 1)[0] > 0.9
                gt_vline_inds2 = gt_vline_inds[:,:,2]
                valid_gt_vline2 = torch.max(gt_vline_inds2, 1)[0] > 0.9
                ind_valid_01 = torch.nonzero(valid_gt_vline0 * valid_gt_vline1)
                ind_valid_12 = torch.nonzero(valid_gt_vline1 * valid_gt_vline2)
                ind_valid_02 = torch.nonzero(valid_gt_vline0 * valid_gt_vline2)

                loss01 = loss12 = loss02 = 0
                if ind_valid_01.size(0)!=0:
                    loss01 = torch.mean((preds[:,0][ind_valid_01] > preds[:,1][ind_valid_01] - LEAST_THRED).float())
                if ind_valid_12.size(0)!=0:
                    loss12 = torch.mean((preds[:,1][ind_valid_12] > preds[:,2][ind_valid_12] - LEAST_THRED).float())
                if ind_valid_02.size(0)!=0:
                    loss02 = torch.mean((preds[:,0][ind_valid_02] > preds[:,2][ind_valid_02] - LEAST_THRED * 2).float())

                loss_order = w1 * loss01 + w2 * loss12 + w3 * loss02
                return loss_classifier, loss_order

def make_vline_loss_evaluator(cfg):    
    # is_focal = True
    # is_focal = False
    
    loss_w = cfg.MODEL.VLINE_HEAD.LOSS_W
    is_focal = cfg.MODEL.VLINE_HEAD.USE_FOCAL_LOSS
    is_simple_bce = cfg.MODEL.VLINE_HEAD.USE_SIMPLE_BCE
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    print("cfg.MODEL.VLINE_HEAD.NUM_BOUDARY: ", cfg.MODEL.VLINE_HEAD.NUM_BOUDARY)
    loss_evaluator = VLineLossComputation(
        matcher,
        # cfg.MODEL.ROI_MASK_HEAD.RESOLUTION,
        cfg.MODEL.VLINE_HEAD.POOLER_RESOLUTION,
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        cfg.MODEL.VLINE_HEAD.BINS,
        loss_w,
        is_focal,
        cfg.MODEL.CLS_AGNOSTIC_VLINE_REG,
        is_simple_bce,
        cfg.MODEL.VLINE_HEAD.NUM_BOUDARY,
    )
    return loss_evaluator
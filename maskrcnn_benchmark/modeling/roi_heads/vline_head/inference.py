import numpy as np
import torch
import torch.nn as nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.utils import cat

class VLinePostProcessor(nn.Module):
    def __init__(self, cls_agnostic_vline_reg):
        super(VLinePostProcessor, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.cls_agnostic_vline_reg = cls_agnostic_vline_reg

    def forward(self, vline_feats, gt_bin, boxes, vps, vert_on, is_roof):

        if not self.cls_agnostic_vline_reg:
            device = vline_feats.device
            labels_pos = cat([proposal.get_field("labels") for proposal in boxes], dim=0)
            assert all(labels_pos > 0)
            map_inds = 3 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2], device=device)

            vline_feats_new = []
            for vline_feat, map_ind in zip(vline_feats, map_inds):
                vline_feats_new.append(vline_feat[:, map_ind][None, :, :])
            vline_feats = torch.cat(vline_feats_new, 0)

        vline_prob = self.softmax(vline_feats)
        valid_half = torch.zeros_like(vline_prob).cuda().float()
        half_size = int(float(valid_half.size(1)) * 0.5) 
        valid_half[:, :half_size, 0] = 1
        valid_half[:, :, 1] = 1
        valid_half[:, half_size:, 2] = 1
        vline_prob = vline_prob * valid_half
        preds = torch.argmax(vline_prob, 1) # Nx2
        preds_score = torch.max(vline_prob, 1)[0]

        gts = torch.argmax(gt_bin, 1) # Nx2, and there can be only 1 nonzero
        gts[torch.sum(gt_bin, 1).float() < 0.1] = -1
        results = []

        preds_score_top, preds_top = torch.topk(vline_prob, k=5, dim=1, largest=True) # B x 5 x 3

        # Instead, use one single box
        box_new = BoxList(boxes[0].bbox, boxes[0].size, mode="xyxy")
        for field in boxes[0].fields():
            box_new.add_field(field, boxes[0].get_field(field))
        if is_roof:
            box_new.add_field("vline_pred_roof", preds)
            box_new.add_field("vline_gt_roof", gts)
            box_new.add_field("vps_roof", vps)
        else:
            if vert_on:
                box_new.add_field("vline_pred_vert_score", preds_score)
                box_new.add_field("vline_pred_vert", preds)
                box_new.add_field("vline_gt_vert", gts)
                box_new.add_field("vp_vert", vps)
                box_new.add_field("line_pred_vert_topk", preds_top)
                box_new.add_field("line_pred_vert_topk_score", preds_score_top)
            else:
                box_new.add_field("vline_pred_score", preds_score)
                box_new.add_field("vline_pred", preds)
                box_new.add_field("vline_gt", gts)
                box_new.add_field("vps", vps)
        assert(len(box_new) == preds.size(0))
        return [box_new]

def make_vline_post_processor(cfg):
    vline_post_processor = VLinePostProcessor(cfg.MODEL.CLS_AGNOSTIC_VLINE_REG)
    return vline_post_processor

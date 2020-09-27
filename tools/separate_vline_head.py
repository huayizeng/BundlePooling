import torch
from torch import nn
import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.vline_head.vline_feature_extractors import make_vline_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.vline_head.VLinePooling import VLinePooling
from maskrcnn_benchmark.modeling.roi_heads.vline_head.VLinePooling2 import VLinePooling2
from maskrcnn_benchmark.modeling.roi_heads.vline_head.VLinePooling3 import VLinePooling3
from maskrcnn_benchmark.modeling.roi_heads.vline_head.VLinePooling4 import VLinePooling4
from maskrcnn_benchmark.modeling.roi_heads.vline_head.vline_utils import indgroupmap_to_indmap_for_many
from maskrcnn_benchmark.modeling.roi_heads.vline_head.loss import make_vline_loss_evaluator
from maskrcnn_benchmark.modeling.roi_heads.vline_head.inference import make_vline_post_processor
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.roi_heads.vline_head.vline_utils import torch2numpy
from maskrcnn_benchmark.layers import Conv2d
from torch.nn import functional as F

import logging
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.logger import setup_logger

# from memory_profiler import profile

class VLineHead(torch.nn.Module):
    def __init__(self, cfg):
        super(VLineHead, self).__init__()
        self.feature_extractor = make_vline_feature_extractor(cfg)
        self.detections_per_img = cfg.MODEL.VLINE_HEAD.DETECTIONS_PER_IMG

        num_backbon_feats_dim = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        num_extract_feats_dim = cfg.MODEL.VLINE_HEAD.NUM_EXTRACT_FEATS_DIM
        num_feats_linear = cfg.MODEL.VLINE_HEAD.NUM_FIRST_LINEAR
        self.use_first_linear = cfg.MODEL.VLINE_HEAD.USE_FIRST_LINEAR
        num_feats_global = 256
        num_linear_another = 512
        
        num_vlines = cfg.MODEL.VLINE_HEAD.BINS if not cfg.MODEL.VLINE_HEAD.USE_FBP else 20
        # print("num_vlines: ", num_vlines)
        self.num_boundary = cfg.MODEL.VLINE_HEAD.NUM_BOUDARY
        self.use_stack = cfg.MODEL.VLINE_HEAD.USE_STACK
        self.use_eye = cfg.MODEL.VLINE_HEAD.USE_EYE
        self.use_global = cfg.MODEL.VLINE_HEAD.USE_GLOBAL
        self.train_nonrf = cfg.MODEL.VLINE_HEAD.TRAIN_NONRF

        if self.use_stack:
            print("using self.use_stack!!!!!!!!!!!!")
            print("using self.use_stack!!!!!!!!!!!!")
            print("using self.use_stack!!!!!!!!!!!!")
            num_stack_out = 256
            if self.use_first_linear:
                self.stack_mean = nn.Linear(num_feats_linear * 2, num_stack_out)
                self.stack_mean_vert = nn.Linear(num_feats_linear * 2, num_stack_out)
                self.stack_max = nn.Linear(num_feats_linear * 2, num_stack_out)
                self.stack_max_vert = nn.Linear(num_feats_linear * 2, num_stack_out)
            else:
                self.stack_mean = nn.Linear(num_extract_feats_dim * 2, num_stack_out)
                self.stack_mean_vert = nn.Linear(num_extract_feats_dim * 2, num_stack_out)
                self.stack_max = nn.Linear(num_extract_feats_dim * 2, num_stack_out)
                self.stack_max_vert = nn.Linear(num_extract_feats_dim * 2, num_stack_out)

        self.use_indsgroupmap = cfg.MODEL.VLINE_HEAD.USE_INDSGROUPMAP
        if self.use_indsgroupmap:
            self.vline_pooling_mean = VLinePooling().apply
            self.vline_pooling_max = VLinePooling4().apply
        else:
            self.vline_pooling_mean = VLinePooling2()
            self.vline_pooling_max = VLinePooling3().apply
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_vline_classes = 1 if cfg.MODEL.CLS_AGNOSTIC_VLINE_REG else num_classes

        if self.use_first_linear:
            d_vline_feats = num_feats_linear * 4
            d_vline_feats2 = num_feats_linear * 2
            self.linear = nn.Linear(num_extract_feats_dim, num_feats_linear)
            self.linear_max = nn.Linear(num_extract_feats_dim, num_feats_linear)
            self.linear_vert = nn.Linear(num_extract_feats_dim, num_feats_linear)
            self.linear_vert_max = nn.Linear(num_extract_feats_dim, num_feats_linear)
        else:
            d_vline_feats = num_extract_feats_dim * 4
            d_vline_feats2 = num_extract_feats_dim * 2

        if self.use_eye: 
            self.eye = torch.tensor(np.eye(num_vlines)).float()
            d_vline_feats+= num_vlines
            d_vline_feats2+= num_vlines
        if self.use_global: 
            d_vline_feats+=num_feats_global
            d_vline_feats2+=num_feats_global
        self.classifier = nn.Linear(d_vline_feats, num_linear_another)
        self.classifier_vert = nn.Linear(d_vline_feats, num_linear_another)
        self.classifier_another = nn.Linear(num_linear_another, self.num_boundary * num_vline_classes)
        self.classifier_vert_another = nn.Linear(num_linear_another, self.num_boundary * num_vline_classes)

        self.classifier_mean = nn.Linear(d_vline_feats2, self.num_boundary * num_vline_classes)
        self.classifier_mean_vert = nn.Linear(d_vline_feats2, self.num_boundary * num_vline_classes)
        self.classifier_max = nn.Linear(d_vline_feats2, self.num_boundary * num_vline_classes)
        self.classifier_max_vert = nn.Linear(d_vline_feats2, self.num_boundary * num_vline_classes)

        self.loss_evaluator = make_vline_loss_evaluator(cfg)
        self.post_processor = make_vline_post_processor(cfg)

        input_size = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        next_feature = input_size
        self.blocks = []

        # TODO(H): If it's possible to remove these convs when extracting feats?
        # Or maybe let's use the one trained on mask?
        layers = cfg.MODEL.VLINE_HEAD.CONV_LAYERS 
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "vp_mask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

        self.logger = logging.getLogger("maskrcnn_benchmark.trainer")
        self.logger.info("Get logger in model")

        num_pooler_resol = cfg.MODEL.VLINE_HEAD.POOLER_RESOLUTION
        # Global: Choice of 1 channel filter
        # self.conv_global = Conv2d(num_backbon_feats_dim, 1, 1)
        # self.linear_global = nn.Linear(num_pooler_resol*num_pooler_resol, num_feats_global)
        
        # Global: Choice of pooling (same as box regre head)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)        
        # self.linear_global = nn.Linear(num_backbon_feats_dim, num_feats_global)
        
        # Global: Choice of pooling (same as shape head)
        ker_size = 8 
        strid = 8
        self.avgpool = nn.AvgPool2d(kernel_size=ker_size, stride=strid)
        num_inputs_global = int(np.floor((num_pooler_resol - ker_size) / strid)) + 1
        self.linear_global = nn.Linear(num_inputs_global*num_inputs_global*num_backbon_feats_dim, num_feats_global)

        # nn.init.normal_(self.linear_global.weight, mean=0, std=0.01)
        # nn.init.constant_(self.linear_global.bias, 0)
        self.softmax = nn.Softmax(dim=1)

    # @profile
    def forward(self, data_all_cuda, training, is_improve=False):
        roi_feats, gt_bin_hor, gt_bin_vert, vline_counts, indgroupmaps, vline_counts_vert, indgroupmaps_vert, save_dict = data_all_cuda

        if training:
            perm = torch.randperm(roi_feats.size(0))
            idx = perm[:64]
        else:
            len_max = roi_feats.size(0)
            idx = torch.tensor(list(range(len_max))).long().cuda()
            print(idx)
        roi_feats = roi_feats[idx]
        # print("before gt_bin_hor.size: ", gt_bin_hor.size())
        if gt_bin_hor is not None:
            gt_bin_hor = gt_bin_hor[idx]
        # print("after gt_bin_hor.size: ", gt_bin_hor.size())
        if gt_bin_vert is not None:
            gt_bin_vert = gt_bin_vert[idx]
        vline_counts = vline_counts[idx]
        # print(indgroupmaps.size())
        indgroupmaps = indgroupmaps[idx]
        # print(indgroupmaps.size())
        vline_counts_vert = vline_counts_vert[idx]
        indgroupmaps_vert = indgroupmaps_vert[idx] 
        for k, v in save_dict.items():
            # print(k)
            if k in ["h_img", "w_img", "fname_base", "inds_quad_roof", "n_neg", "inds_beveled"]:
                continue
            if save_dict[k] is None:
                # For F-BP and Bi-BP, vp doesn't exist
                continue

            save_dict[k] = save_dict[k][idx]

        if roi_feats.size(0) == 0:
            print("No roi_feats!!!!")
            return {}

        ### Compute global feats (Before conv of vlines)
        # Conv1 global
        # roi_feats_1channel = F.relu(self.conv_global(roi_feats))
        # Pool globel
        num_vlines = indgroupmaps.size(3)
        if self.use_global:
            roi_feats_1channel = self.avgpool(roi_feats)
            global_feats = F.relu(self.linear_global(roi_feats_1channel.view(roi_feats_1channel.size(0), -1)))
            global_feats = torch.unsqueeze(global_feats, 1).expand(global_feats.size(0), num_vlines, global_feats.size(1))

        for layer_name in self.blocks:
            roi_feats = F.relu(getattr(self, layer_name)(roi_feats))

        valid_vecs = vline_counts != -1
        assert valid_vecs.all().item()
        # valid_vecs = valid_vecs.unsqueeze(2).expand(valid_vecs.size(0), valid_vecs.size(1), 3)

        valid_vecs_vert = vline_counts_vert != -1
        assert valid_vecs.all().item()
        # valid_vecs_vert = valid_vecs_vert.unsqueeze(2).expand(valid_vecs_vert.size(0), valid_vecs_vert.size(1), 3)

        ### Compute ind map
        indmaps, valid_maps = indgroupmap_to_indmap_for_many(indgroupmaps)
        indmaps_vert, valid_maps_vert = indgroupmap_to_indmap_for_many(indgroupmaps_vert)

        ### Mean x hor        
        if self.use_indsgroupmap:
            vline_feats = self.vline_pooling_mean(roi_feats, indgroupmaps, vline_counts)
        else:
            vline_feats = self.vline_pooling_mean(roi_feats, indmaps, vline_counts, valid_maps)
        vline_feats = torch.transpose(vline_feats, 1, 2)
        if self.use_first_linear:
            vline_feats = self.linear(vline_feats)
        ### Mean x vert
        if self.use_indsgroupmap:
            vline_feats_vert = self.vline_pooling_mean(roi_feats, indgroupmaps_vert, vline_counts_vert)
        else:
            vline_feats_vert = self.vline_pooling_mean(roi_feats, indmaps_vert, vline_counts_vert, valid_maps_vert)
        vline_feats_vert = torch.transpose(vline_feats_vert, 1, 2)
        if self.use_first_linear:
            vline_feats_vert = self.linear_vert(vline_feats_vert)

        ### Max x hor
        if self.use_indsgroupmap:
            vline_feats_max = self.vline_pooling_max(roi_feats, indgroupmaps, vline_counts, valid_maps)
        else:
            vline_feats_max = self.vline_pooling_max(roi_feats, indmaps, vline_counts, valid_maps)
        vline_feats_max = torch.transpose(vline_feats_max, 1, 2)
        if self.use_first_linear:
            vline_feats_max = self.linear_max(vline_feats_max)

        ### Max x vert
        if self.use_indsgroupmap:
            vline_feats_vert_max = self.vline_pooling_max(roi_feats, indgroupmaps_vert, vline_counts_vert, valid_maps_vert)
        else:
            vline_feats_vert_max = self.vline_pooling_max(roi_feats, indmaps_vert, vline_counts_vert, valid_maps_vert)
        vline_feats_vert_max = torch.transpose(vline_feats_vert_max, 1, 2)
        if self.use_first_linear:
            vline_feats_vert_max = self.linear_vert_max(vline_feats_vert_max)
        # vline_feats_vert = vline_feats_vert + vline_feats_vert_max

        ### Concate mean, max, global
        if self.use_eye:
            eyes = torch.unsqueeze(self.eye, 0).expand(roi_feats.size(0), num_vlines, num_vlines).cuda()
        # print(eyes[0,:,:])
        # print(eyes[2,:,:])
        vline_feats_pool_max = torch.max(vline_feats_max, 1)[0]
        vline_feats_pool_mean = torch.mean(vline_feats, 1)
        vline_feats_pool_max = torch.unsqueeze(vline_feats_pool_max, 1).expand(vline_feats_pool_max.size(0), num_vlines, vline_feats_pool_max.size(1))
        vline_feats_pool_mean = torch.unsqueeze(vline_feats_pool_mean, 1).expand(vline_feats_pool_mean.size(0), num_vlines, vline_feats_pool_mean.size(1))

        if self.use_eye and self.use_global:
            vline_feats_all = torch.cat((eyes, vline_feats, vline_feats_max, vline_feats_pool_mean, vline_feats_pool_max, global_feats), 2)
            vline_feats_mean = torch.cat((eyes, vline_feats, vline_feats_pool_mean, global_feats), 2)
            vline_feats_max = torch.cat((eyes, vline_feats_max, vline_feats_pool_max, global_feats), 2)
        else:
            vline_feats_all = torch.cat((vline_feats, vline_feats_max, vline_feats_pool_mean, vline_feats_pool_max), 2)
            vline_feats_mean = torch.cat((vline_feats, vline_feats_pool_mean), 2)
            vline_feats_max = torch.cat((vline_feats_max, vline_feats_pool_max), 2)

        vline_feats_vert_pool_max = torch.max(vline_feats_vert_max, 1)[0]
        vline_feats_vert_pool_mean = torch.mean(vline_feats_vert, 1)
        vline_feats_vert_pool_max = torch.unsqueeze(vline_feats_vert_pool_max, 1).expand(vline_feats_vert_pool_max.size(0), num_vlines, vline_feats_vert_pool_max.size(1))
        vline_feats_vert_pool_mean = torch.unsqueeze(vline_feats_vert_pool_mean, 1).expand(vline_feats_vert_pool_mean.size(0), num_vlines, vline_feats_vert_pool_mean.size(1))        
        if self.use_eye and self.use_global:
            vline_feats_vert_all = torch.cat((eyes, vline_feats_vert, vline_feats_vert_max, vline_feats_vert_pool_mean, vline_feats_vert_pool_max, global_feats), 2)
            vline_feats_vert_mean = torch.cat((eyes, vline_feats_vert, vline_feats_vert_pool_mean, global_feats), 2)
            vline_feats_vert_max = torch.cat((eyes, vline_feats_vert_max, vline_feats_vert_pool_max, global_feats), 2)
        else:                
            vline_feats_vert_all = torch.cat((vline_feats_vert, vline_feats_vert_max, vline_feats_vert_pool_mean, vline_feats_vert_pool_max), 2)
            vline_feats_vert_mean = torch.cat((vline_feats_vert, vline_feats_vert_pool_mean), 2)
            vline_feats_vert_max = torch.cat((vline_feats_vert_max, vline_feats_vert_pool_max), 2)

        vline_feats_all = F.relu(self.classifier(vline_feats_all))
        vline_feats_vert_all = F.relu(self.classifier_vert(vline_feats_vert_all))
        vline_feats_all = self.classifier_another(vline_feats_all)
        vline_feats_vert_all = self.classifier_vert_another(vline_feats_vert_all)
        
        vline_feats_mean = self.classifier_mean(vline_feats_vert_mean)
        vline_feats_vert_mean = self.classifier_mean_vert(vline_feats_vert_mean)
        vline_feats_max = self.classifier_max(vline_feats_max)
        vline_feats_vert_max = self.classifier_max_vert(vline_feats_vert_max)

        if not self.training and is_improve:
            vline_prob = self.softmax(vline_feats_all)
            preds = torch.argmax(vline_prob, 1) # Nx2
            vline_prob_vert = self.softmax(vline_feats_vert_all)
            preds_vert = torch.argmax(vline_prob_vert, 1) # Nx2
            return preds, preds_vert

        ### Creating boxlist for inference and loss
        box_new = BoxList(save_dict["boxes"], [save_dict["w_img"], save_dict["h_img"]], mode="xyxy")
        box_new.add_field("labels", save_dict["labels"])
        if "shapes" in save_dict.keys():
            box_new.add_field("shapes", save_dict["shapes"])

        # kept = list(filter(lambda e: e[0]==1 and e[1]==5, zip(save_dict["shapes"], save_dict["labels"], range(len(save_dict["labels"])))))
        if "shapes" in save_dict.keys():
            kept = list(filter(lambda e: e[1]==5, zip(save_dict["shapes"], save_dict["labels"], range(len(save_dict["labels"])))))
            inds_quad_roof = [k[2] for k in kept]
            box_new_roof = box_new[inds_quad_roof]
            vline_feats_all_roof = vline_feats_all[inds_quad_roof]
            vline_feats_mean_roof = vline_feats_mean[inds_quad_roof]
            vline_feats_max_roof = vline_feats_max[inds_quad_roof]
            gt_bin_hor_roof = gt_bin_hor[inds_quad_roof]
            vps_roof = save_dict["vps"][inds_quad_roof]
            
            kept = list(filter(lambda e: e[0]!=5, zip(save_dict["labels"], range(len(save_dict["labels"])))))
            inds_non_roof = [k[1] for k in kept]
            box_new = box_new[inds_non_roof]
            vline_feats_all = vline_feats_all[inds_non_roof]
            vline_feats_mean = vline_feats_mean[inds_non_roof]
            vline_feats_max = vline_feats_max[inds_non_roof]
            vline_feats_vert_all = vline_feats_vert_all[inds_non_roof]
            vline_feats_vert_mean = vline_feats_vert_mean[inds_non_roof]
            vline_feats_vert_max = vline_feats_vert_max[inds_non_roof]
            gt_bin_hor = gt_bin_hor[inds_non_roof]
            gt_bin_vert = gt_bin_vert[inds_non_roof]
            save_dict["vps"] = save_dict["vps"][inds_non_roof]
            save_dict["vp_vert"] = save_dict["vp_vert"][inds_non_roof]

        if not self.training:
            if len(inds_non_roof) !=0:
                result = self.post_processor(vline_feats_all, gt_bin_hor, [box_new], save_dict["vps"], vert_on=False, is_roof=False)
                result = self.post_processor(vline_feats_vert_all, gt_bin_vert, result, save_dict["vp_vert"], vert_on=True, is_roof=False)
            else:
                result = None
            # result = self.post_processor(vline_feats_mean, gt_bin_hor, [box_new], save_dict["vps"], vert_on=False, is_roof=False)
            # result = self.post_processor(vline_feats_vert_mean, gt_bin_vert, result, save_dict["vp_vert"], vert_on=True, is_roof=False)
            # result = self.post_processor(vline_feats_max, gt_bin_hor, [box_new], save_dict["vps"], vert_on=False, is_roof=False)
            # result = self.post_processor(vline_feats_vert_max, gt_bin_vert, result, save_dict["vp_vert"], vert_on=True, is_roof=False)
            if len(box_new_roof) != 0:
                # result_roof = self.post_processor(vline_feats_all_roof, gt_bin_hor_roof, [box_new_roof], vps_roof, vert_on=False, is_roof=True)
                # result_roof = self.post_processor(vline_feats_mean_roof, gt_bin_hor_roof, [box_new_roof], vps_roof, vert_on=False, is_roof=True)
                result_roof = self.post_processor(vline_feats_max_roof, gt_bin_hor_roof, [box_new_roof], vps_roof, vert_on=False, is_roof=True)
            else:
                result_roof = None
            return result, result_roof

        if len(box_new) == 0:
            print("This should happen very rarely")
            return {}

        # print("gt_bin_hor.size: ", gt_bin_hor.size())
        if self.train_nonrf:
            loss_vline, loss_order = self.loss_evaluator(
                [box_new], vline_feats_all, gt_bin_hor, is_vert=False
            )
            loss_dict = dict(loss_vline=loss_vline)
            if loss_order:
                loss_dict.update(loss_order=loss_order)

            loss_vline_vert, loss_order_vert = self.loss_evaluator(
                [box_new], vline_feats_vert_all, gt_bin_vert, is_vert=True
            )
            loss_dict.update(dict(loss_vline_vert=loss_vline_vert))
            if loss_order_vert:
                loss_dict.update(dict(loss_order_vert=loss_order_vert))

            ######## mean only
            loss_vline_mean, loss_order_mean = self.loss_evaluator(
                [box_new], vline_feats_mean, gt_bin_hor, is_vert=False
            )
            loss_dict.update(loss_vline_mean=loss_vline_mean)
            if loss_order_mean:
                loss_dict.update(loss_order_mean=loss_order_mean)

            loss_vline_vert_mean, loss_order_vert_mean = self.loss_evaluator(
                [box_new], vline_feats_vert_mean, gt_bin_vert, is_vert=True
            )
            loss_dict.update(dict(loss_vline_vert_mean=loss_vline_vert_mean))
            if loss_order_vert_mean:
                loss_dict.update(dict(loss_order_vert_mean=loss_order_vert_mean))

            ######## max only
            loss_vline_max, loss_order_max = self.loss_evaluator(
                [box_new], vline_feats_max, gt_bin_hor, is_vert=False
            )
            loss_dict.update(loss_vline_max=loss_vline_max)
            if loss_order_max:
                loss_dict.update(loss_order_max=loss_order_max)
            loss_vline_vert_max, loss_order_vert_max = self.loss_evaluator(
                [box_new], vline_feats_vert_max, gt_bin_vert, is_vert=True
            )
            loss_dict.update(dict(loss_vline_vert_max=loss_vline_vert_max))
            if loss_order_vert_max:
                loss_dict.update(dict(loss_order_vert_max=loss_order_vert_max))
            return loss_dict
        
        else:
            ######## roofing
            if len(box_new_roof) == 0:
                return {}

            loss_vline_roof, loss_order_roof = self.loss_evaluator(
                [box_new_roof], vline_feats_all_roof, gt_bin_hor_roof, is_vert=False
            )
            loss_dict= dict(loss_vline_roof=loss_vline_roof)
            loss_dict.update(loss_order_roof=loss_order_roof)

            loss_vline_mean_roof, loss_order_mean_roof = self.loss_evaluator(
                [box_new_roof], vline_feats_mean_roof, gt_bin_hor_roof, is_vert=False
            )
            loss_dict.update(loss_vline_mean_roof=loss_vline_mean_roof)
            loss_dict.update(loss_order_mean_roof=loss_order_mean_roof)

            loss_vline_max_roof, loss_order_max_roof = self.loss_evaluator(
                [box_new_roof], vline_feats_max_roof, gt_bin_hor_roof, is_vert=False
            )
            loss_dict.update(loss_vline_max_roof=loss_vline_max_roof)
            loss_dict.update(loss_order_max_roof=loss_order_max_roof)

            # memReport()
            return loss_dict
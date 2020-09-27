import torch
import numpy as np
import torch.nn as nn

class VLinePooling2(nn.Module):
    def forward(self, input, indmap, output_count, valid_maps):
        B = input.size(0)
        C = input.size(1)
        H = input.size(2)
        W = input.size(3)
        L = output_count.size(1)
        output_count = output_count.unsqueeze(1).expand(B, C, L)

        arange_b, arange_c = list(range(B)), list(range(C))
        ind_c, ind_b = np.meshgrid(arange_c, arange_b)
        ind_b = torch.from_numpy(ind_b).int().cuda()
        ind_c = torch.from_numpy(ind_c).int().cuda()
        # exit(0)
        indmap = indmap.unsqueeze(1).expand(B, C, H, W)
        valid_maps = valid_maps.unsqueeze(1).expand(B, C, H, W)
        input = input * valid_maps.float()

        inds_accum = (ind_b * C + ind_c) * L
        inds_accum = inds_accum.unsqueeze(2).expand(B, C, H)
        inds_accum = inds_accum.unsqueeze(3).expand(B, C, H, W)
        inds_accum = inds_accum + indmap
        output = torch.zeros([B, C, L]).cuda()
        output.put_(inds_accum.long(), input, accumulate=True)
        # exit(0)
        output_mean = output / output_count.float()
        return output_mean

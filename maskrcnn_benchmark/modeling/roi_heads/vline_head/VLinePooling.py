import torch
import numpy as np
import torch.nn as nn

import gc
from functools import reduce
import operator

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            # print(type(obj), obj.size())
            print(reduce(operator.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
    print("----------------------------")

class VLinePooling(torch.autograd.Function):
    @staticmethod
    def forward(self, input, indgroupmap, output_count):
        B = input.size(0)
        C = input.size(1)
        H = input.size(2)
        W = input.size(3)
        L = indgroupmap.size(3) # indgroupmap.shape = BHWL

        arange_b, arange_c = list(range(B)), list(range(C))
        ind_c, ind_b = np.meshgrid(arange_c, arange_b)
        ind_b = torch.from_numpy(ind_b).int().cuda()
        ind_c = torch.from_numpy(ind_c).int().cuda()
        # ind_b = torch.from_numpy(ind_b).int()
        # ind_c = torch.from_numpy(ind_c).int()

        output = torch.zeros([B, C, L]).cuda()
        for i in range(L):
            indgroupmap_temp = indgroupmap[:,:,:,i].unsqueeze(1).expand(B, C, H, W)
            input_temp_mul = input * indgroupmap_temp.float()
            inds_accum = ind_b * C + ind_c
            inds_accum = inds_accum.unsqueeze(2).expand(B, C, H)
            inds_accum = inds_accum.unsqueeze(3).expand(B, C, H, W)
            output_temp = output[:,:,i]
            output_temp.put_(inds_accum.long(), input_temp_mul, accumulate=True)

        output_count = output_count.unsqueeze(1).expand(B, C, L).float()
        output_mean = output / output_count
        self.save_for_backward(indgroupmap, output_count)
        return output_mean

    @staticmethod
    def backward(self, grad_output):
        # print("Start back for vline")
        indgroupmap, output_count = self.saved_tensors
        # NOTE(H): No worries about invalid, as valid_vecs in 
        # vline_head.py (= ~invalid) will serve as a mask, making the
        # related loss to be zero. 
        B = output_count.size(0)
        C = output_count.size(1)
        H = indgroupmap.size(1)
        W = indgroupmap.size(2)
        L = output_count.size(2) # indgroupmap.shape = BHWL
        grad_output = grad_output / output_count.float()
        grad_output_temp_mul = torch.zeros([B, C, H, W]).cuda()
        for i in range(L):
            grad_output_temp = grad_output[:,:,i].unsqueeze(2).expand(B, C, H)
            grad_output_temp = grad_output_temp.unsqueeze(3).expand(B, C, H, W)
            indgroupmap_temp = indgroupmap[:,:,:,i].unsqueeze(1).expand(B, C, H, W)
            grad_output_temp_mul += grad_output_temp * indgroupmap_temp.float()
        grad_input = grad_output_temp_mul
        # memReport()
        return grad_input, None,  None

# class VLinePooling(nn.Module):
#     def forward(self, input, indgroupmap):
#         B = input.size(0)
#         C = input.size(1)
#         H = input.size(2)
#         W = input.size(3)
#         L = indgroupmap.size(3) # indgroupmap.shape = BHWL

#         arange_b, arange_c = list(range(B)), list(range(C))
#         ind_b, ind_c = np.meshgrid(arange_b, arange_c)
#         # ind_b = torch.from_numpy(ind_b).int().cuda()
#         # ind_c = torch.from_numpy(ind_c).int().cuda()
#         ind_b = torch.from_numpy(ind_b).int()
#         ind_c = torch.from_numpy(ind_c).int()

#         indgroupmap_temp = indgroupmap.unsqueeze(1).expand(B, C, H, W, L)
#         input_temp = input.unsqueeze(4).expand(B, C, H, W, L)
#         input_temp_mul = input_temp * indgroupmap_temp.float()
#         # inds = torch.Tensor(list(range(L))).int().cuda()
#         inds = torch.Tensor(list(range(L))).int()
#         inds_temp = inds.repeat(B, C, H, W, 1)

#         inds_accum = ind_b * C + ind_c
#         inds_accum = inds_accum.repeat(1, 1, H, W, L) # BCHWL
#         inds_accum += inds_temp

#         # output = torch.zeros([B, C, L]).cuda()
#         # output_count = torch.zeros([B, C, L]).cuda()
#         output = torch.zeros([B, C, L])
#         output_count = torch.zeros([B, C, L])
#         output.put_(inds_accum.long(), input_temp_mul, accumulate=True)
#         output_count.put_(inds_accum.long(), indgroupmap_temp.float(), accumulate=True)
#         output_mean = output / output_count
#         return output_mean



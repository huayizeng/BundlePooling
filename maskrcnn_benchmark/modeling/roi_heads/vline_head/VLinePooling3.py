import torch
import numpy as np
import torch.nn as nn

# class VLinePooling3(nn.Module):
#     def forward(self, input, indmap, output_count, valid_maps):
#         B = input.size(0)
#         C = input.size(1)
#         H = input.size(2)
#         W = input.size(3)
#         # print("output_count.size(): ", output_count.size())
#         L = output_count.size(1)
#         print("B, C, H, W, L: ", B, C, H, W, L)
#         output_count = output_count.unsqueeze(1).expand(B, C, L)
#         # print(output_count.requires_grad)

#         arange_b, arange_c = list(range(B)), list(range(C))
#         ind_c, ind_b = np.meshgrid(arange_c, arange_b)
#         ind_b = torch.from_numpy(ind_b).int().cuda()
#         ind_c = torch.from_numpy(ind_c).int().cuda()
#         # print(input.size())
#         # print(indmap.size())
#         # print(indmap)
#         # print(indmap)
#         # exit(0)
#         indmap = indmap.unsqueeze(1).expand(B, C, H, W)
#         # input_temp_mul = input * indmap.float()

#         # inds_accum = (ind_b * C + ind_c) * L
#         # inds_accum = inds_accum.unsqueeze(2).expand(B, C, H)
#         # inds_accum = inds_accum.unsqueeze(3).expand(B, C, H, W)
#         # inds_accum = inds_accum + indmap
#         output_max = torch.zeros([B, C, L]).cuda()

#         valid_maps = valid_maps.unsqueeze(1).expand(B, C, H, W)
#         input = input * valid_maps.float()

#         for i in range(L):
#             mask = indmap != i
#             # print("----")
#             # print(mask)
#             # print("mask.size(): ", mask.size())
#             # print("input.size(): ", input.size())
#             input_temp = input.clone()
#             input_temp[mask] = -999999
#             # print(input_temp)
#             max_temp = torch.max(input_temp, 2)[0]
#             # print("max_temp: ", max_temp)
#             output_max[:,:,i] = max_temp
#             # print("max_temp.size(): ", max_temp.size())
#             # exit(0)

#         # output.put_(inds_accum.long(), input, accumulate=True)
#         # print(output.is_cuda)
#         # print(output_count.is_cuda)
#         # exit(0)
#         return output_max

class VLinePooling3(torch.autograd.Function):
    @staticmethod
    def forward(self, input, indmap, output_count, valid_maps):
        B = input.size(0)
        C = input.size(1)
        H = input.size(2)
        W = input.size(3)
        L = output_count.size(1)
        # print("B, C, H, W, L: ", B, C, H, W, L)

        arange_b, arange_c = list(range(B)), list(range(C))
        ind_c, ind_b = np.meshgrid(arange_c, arange_b)
        ind_b = torch.from_numpy(ind_b).int().cuda()
        ind_c = torch.from_numpy(ind_c).int().cuda()
        indmap = indmap.unsqueeze(1).expand(B, C, H, W)
        output_max = torch.zeros([B, C, L]).cuda()

        valid_maps = valid_maps.unsqueeze(1).expand(B, C, H, W).byte()
        # input = input * valid_maps.float()
        argmaxs = torch.zeros([B, C, L]).cuda().int()
        for i in range(L):
            mask = indmap != i
            input_temp = input.clone()
            input_temp[mask] = -torch.Tensor(float("Inf")).cuda()
            input_temp[~valid_maps] = -torch.Tensor(float("Inf")).cuda()
            input_temp = input_temp.view(B, C, -1)
            max_temp = torch.max(input_temp, 2)[0]
            argmax_temp = torch.argmax(input_temp, 2)
            output_max[:,:,i] = max_temp
            argmaxs[:, :, i] = argmax_temp.int()

        self.save_for_backward(argmaxs, ind_b, ind_c, torch.tensor(H).cuda(), torch.tensor(W).cuda())
        return output_max

    @staticmethod
    def backward(self, grad_output):
        argmaxs, ind_b, ind_c, H, W = self.saved_tensors
        B = argmaxs.size(0)
        C = argmaxs.size(1)
        L = argmaxs.size(2)
        inds_accum = (ind_b * C + ind_c) * H * W
        inds_accum = inds_accum.unsqueeze(2).expand(B, C, L)
        inds_accum = inds_accum + argmaxs

        grad_input = torch.zeros([B, C, H, W]).cuda()
        grad_input.put_(inds_accum.long(), grad_output, accumulate=True)
        return grad_input, None, None, None
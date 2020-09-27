import torch
import numpy as np
import torch.nn as nn

import gc
# from torchsummary import summary
from functools import reduce
import operator

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            # print(type(obj), obj.size())
            print(reduce(operator.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
    print("----------------------------")
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
    #     except:
    #         pass
    # print("----------------------------")

class VLinePooling4(torch.autograd.Function):
    @staticmethod
    def forward(self, input, indgroupmap, output_count, valid_maps):
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
        # indgroupmap = indgroupmap.unsqueeze(1).expand(B, C, H, W, L)
        # print("indgroupmap.shape: ", indgroupmap.shape)
        output_max = torch.zeros([B, C, L]).cuda()

        # todo(h): here the valid_maps could be removed
        valid_maps = valid_maps.unsqueeze(1).expand(B, C, H, W).byte()
        # input = input * valid_maps.float()
        argmaxs = torch.zeros([B, C, L]).cuda().int()
        # memReport()  
        for i in range(L):
            # print("i: ", i)
            indgroupmap_temp = indgroupmap[:,:,:,i].unsqueeze(1).expand(B, C, H, W)
            mask = ~indgroupmap_temp

            input_temp = input.clone()
            input_temp[mask] = -999999
            input_temp[~valid_maps] = -999999
            # if i == 0:
            #     print(input_temp)
            input_temp = input_temp.view(B, C, -1)
            max_temp = torch.max(input_temp, 2)[0]
            argmax_temp = torch.argmax(input_temp, 2)
            # print(argmax_temp)
            # print(argmax_temp.size())
            # Got the argmax for back-prop
            output_max[:,:,i] = max_temp
            argmaxs[:, :, i] = argmax_temp.int()
      
        self.save_for_backward(argmaxs, ind_b, ind_c, torch.tensor(H).cuda(), torch.tensor(W).cuda())
        return output_max

    @staticmethod
    def backward(self, grad_output):
        # print("----")
        argmaxs, ind_b, ind_c, H, W = self.saved_tensors
        B = argmaxs.size(0)
        C = argmaxs.size(1)
        L = argmaxs.size(2)
        # print(grad_output.size())
        # print(argmaxs.size())
        inds_accum = (ind_b * C + ind_c) * H * W
        inds_accum = inds_accum.unsqueeze(2).expand(B, C, L)
        inds_accum = inds_accum + argmaxs

        grad_input = torch.zeros([B, C, H, W]).cuda()
        # print(inds_accum)
        grad_input.put_(inds_accum.long(), grad_output, accumulate=True)
        # print(grad_input)
        # memReport()
        return grad_input, None, None, None

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLinePooling4()
    # summary does not work
    # summary(model, [(10, 256, 28, 28), (10, 28, 28, 40), (10, 40), (10, 28, 28)])
    # pass
    # # summary(VLinePooling4, )
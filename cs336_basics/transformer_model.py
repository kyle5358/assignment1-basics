import math
import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None,dtype=None,):
        super().__init__()
        #权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features,device=device, dtype=dtype))
        #计算sigma
        sigma = math.sqrt(2.0/(in_features +out_features))
        #截断
        nn.init.trunc_normal_(self.weight, mean=0,std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #einsum线性变换
        return einsum(x,self.weight,"... d_in,d_out d_in->... d_out")
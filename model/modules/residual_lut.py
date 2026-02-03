
import torch
import torch.nn as nn
from torch import Tensor

class ResidualLUT(nn.Module):
    def __init__(self, dim=33):
        super().__init__()
        # 初始化残差 LUT，形状为 (1, 3, D, D, D)
        # 初始值为 0，表示不改变原图
        self.lut_residual = nn.Parameter(torch.zeros(1, 3, dim, dim, dim))
        
        # 使用 Xavier 初始化微小的随机扰动，打破对称性，有助于逃离局部最优
        # 但要保持非常小，以免破坏初始图像
        nn.init.uniform_(self.lut_residual, -1e-5, 1e-5)

    def forward(self, identity_lut):
        """
        :param identity_lut: 基础的 Identity LUT (1, 3, D, D, D)
        """
        # 最终 LUT = Identity + Residual
        return identity_lut + self.lut_residual

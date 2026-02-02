import torch
import torch.nn as nn
import torch.nn.functional as F

class LUTApplier(nn.Module):
    """
    将 3D LUT 应用到图像上，使用三线性插值 (Trilinear Interpolation)
    """
    
    def __init__(self, n_vertices=33):
        super(LUTApplier, self).__init__()
        self.n_vertices = n_vertices
    
    def forward(self, image: torch.Tensor, lut):
        """
        Args:
            image: (B, 3, H, W) - 输入图像，像素值范围 [0, 1]
            lut: (B, 3, 33, 33, 33) - 预测的 3D LUT
        Returns:
            output: (B, 3, H, W) - 应用 LUT 后的图像
        """
        B, C, H, W = image.shape
        
        # 1. 将图像像素值从 [0,1] 映射到 LUT 索引空间 [-1,1] (grid_sample 要求)
        # image: (B, 3, H, W) -> (B, H, W, 3)
        image = image.permute(0, 2, 3, 1)
        
        # 归一化到 [-1, 1]（grid_sample 的坐标范围）
        grid = image * 2 - 1  # [0,1] -> [-1,1]
        
        # 2. 调整维度以适配 grid_sample
        # grid_sample 要求 5D 输入: (B, C, D, H, W) 和 grid: (B, D_out, H_out, W_out, 3)
        # 我们把图像的每个像素当作一个 3D 坐标去查 LUT
        
        # grid: (B, H, W, 3) -> (B, H*W, 1, 1, 3)
        grid = grid.view(B, -1, 1, 1, 3)
        
        # lut: (B, 3, 33, 33, 33) 已经是正确格式
        
        # 3. 使用 grid_sample 进行三线性插值
        output = F.grid_sample(
            lut, 
            grid,
            mode='bilinear', 
            padding_mode='border',
            align_corners=True 
        )
        
        # 4. 还原形状: (B, 3, H*W, 1, 1) -> (B, 3, H, W)
        output = output.view(B, C, H, W)
        
        return output
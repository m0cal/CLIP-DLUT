import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableChineseCLIPProcessor(nn.Module):
    def __init__(self, crop_size=336):
        super().__init__()
        self.crop_size = crop_size
        
        # Chinese-CLIP/OpenAI CLIP 标配的均值和方差
        # 必须注册为 buffer，这样它会随模型移动到 GPU，但不会计算梯度
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Args:
            x: 形状为 [B, 3, H, W] 的 Tensor，取值范围 [0, 1]
               注意：必须先确保 x.requires_grad = True
        Returns:
            processed_x: 形状为 [B, 3, 336, 336] 的标准化后的 Tensor
        """
        B, C, H, W = x.shape
        
        # 1. Resize (缩放短边到 336)
        # 逻辑：找出短边，计算缩放比例
        scale = self.crop_size / min(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        
        # 使用双三次插值 (Bicubic)，这是 CLIP 的标准设置，且是可导的
        x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
        
        # 2. Center Crop (中心裁剪 336x336)
        start_y = (new_h - self.crop_size) // 2
        start_x = (new_w - self.crop_size) // 2
        x = x[:, :, start_y:start_y+self.crop_size, start_x:start_x+self.crop_size]
        
        # 3. Normalize (归一化)
        # x 此时应在 [0, 1] 之间，如果输入是 0-255，请先除以 255
        x = (x - self.mean) / self.std
        
        return x
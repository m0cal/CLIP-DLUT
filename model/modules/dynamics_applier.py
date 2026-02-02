import torch 
import torch.nn as nn
from torch import Tensor
from .gradient_predictor import GradientPredictor

class DynamicsApplier(nn.Module):
    def __init__(self, epsilon=1e-3, use_noise=False):
        super().__init__()
        self.epsilon = epsilon
        self.use_noise = use_noise

    def forward(self, lut: Tensor, predictor: GradientPredictor):
        b, c, d, h, w = lut.shape
        # 将 LUT 展平为顶点列表 [N, 3]
        vertices = lut.permute(0, 2, 3, 4, 1).reshape(-1, 3)

        # 1. 预测梯度场 (保持计算图)
        gradient = predictor(vertices)

        # 2. 更新顶点 (去掉 .detach())
        # 注意：这里是 Predictor 训练的关键，gradient 必须参与计算
        new_vertices = vertices + (self.epsilon / 2.0) * gradient

        # 3. 添加噪声 (如果是在做采样或增加鲁棒性)
        if self.use_noise:
            # 噪声不需要梯度，使用 randn_like 即可
            noise = torch.randn_like(new_vertices)
            # 这里的 sqrt(epsilon) 是 Langevin Dynamics 的标准写法
            new_vertices = new_vertices + torch.sqrt(torch.as_tensor(self.epsilon)) * noise

        # 4. 限制范围 (使用非原地操作)
        new_vertices = torch.clamp(new_vertices, 0, 1)

        # 5. 还原形状
        new_lut = new_vertices.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)

        return new_lut
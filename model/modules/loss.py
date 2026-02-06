from .clip_loss import CLIPLoss
import torch
from torch import Tensor
import torch.nn as nn
from dataclasses import dataclass

def get_color_volume(image):
    """
    计算图像在 LAB 色彩空间中的统计体积。
    在感知均匀的 LAB 空间中计算体积，能更准确地衡量“色彩丰富度”。
    """
    # 转换为 LAB 空间
    lab_image = rgb_to_lab(image)
    
    # 归一化 LAB 到 [0, 1] 左右的尺度，方便计算协方差
    # L: 0~100 -> 0~1
    # a, b: -128~127 -> -1~1
    lab_image[:, 0, :, :] = lab_image[:, 0, :, :] / 100.0
    lab_image[:, 1, :, :] = lab_image[:, 1, :, :] / 128.0
    lab_image[:, 2, :, :] = lab_image[:, 2, :, :] / 128.0

    b, c, h, w = lab_image.shape
    pixels = lab_image.view(b, c, -1)
    
    # 中心化
    centered = pixels - pixels.mean(dim=2, keepdim=True)
    # 计算协方差矩阵 [B, 3, 3]
    cov = torch.bmm(centered, centered.transpose(1, 2)) / (h * w - 1)
    # 添加扰动，防止行列式为0导致反向传播梯度为 nan
    cov = cov + torch.eye(3, device=image.device).unsqueeze(0) * 1e-6
    # 计算行列式
    vol = torch.det(cov).abs()
    return vol

def rgb_to_lab(image: Tensor) -> Tensor:
    """
    将 RGB 转换到 LAB 空间 (可微分近似)
    输入: [B, 3, H, W], 数值 [0, 1]
    输出: [B, 3, H, W], L [0, 100], a [-128, 127], b [-128, 127]
    """
    # 1. Linearize RGB (Gamma Correction)
    mask = image > 0.04045
    image = torch.where(mask, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)

    # 2. RGB to XYZ
    r, g, b = image[:, 0, ...], image[:, 1, ...], image[:, 2, ...]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize for D65 white point
    x = x / 0.95047
    z = z / 1.08883

    # 3. XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3

    def f(t):
        # 使用平滑的 softplus 或其他近似替代硬阈值判断，或者确保不出现梯度 NaN
        # 这里为了数值稳定性，直接加上一个极小值避免 0
        t = torch.clamp(t, min=1e-4) # 避免 t <= 0 导致 power(1/3) 梯度爆炸
        
        mask = t > epsilon
        return torch.where(mask, torch.pow(t, 1/3), (kappa * t + 16) / 116)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return torch.stack([L, a, b], dim=1)


def monotonicity_loss(lut):
    # lut shape: (3, Dim, Dim, Dim)
    # 1. 单调性约束：惩罚负梯度 (防止颜色反转)
    dx = lut[:, 1:, :, :] - lut[:, :-1, :, :]
    dy = lut[:, :, 1:, :] - lut[:, :, :-1, :]
    dz = lut[:, :, :, 1:] - lut[:, :, :, :-1]
    
    loss_mono = torch.relu(-dx).mean() + torch.relu(-dy).mean() + torch.relu(-dz).mean() * 0.1
    
    # 2. 平滑性约束（修正版）：惩罚二阶差分 (Curvature)，鼓励线性变化，防止锯齿
    # ddx = dx[i+1] - dx[i]
    ddx = dx[:, 1:, :, :] - dx[:, :-1, :, :]
    ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    ddz = dz[:, :, :, 1:] - dz[:, :, :, :-1]
    
    # 使用 L2 范数惩罚曲率
    # 降低平滑权重，避免它过度限制 LUT 的变形能力，从而阻碍 CLIP 优化
    loss_smooth = (ddx.square().mean() + ddy.square().mean() + ddz.square().mean()) * 1.0

    return loss_mono, loss_smooth

@dataclass
class LossWeights:
    clip: float
    clip_c: float
    mono: float
    smoothness: float
    color_vol: float
    color_shift: float
    color_eigen: float

class AllLoss(nn.Module):
    def __init__(self, 
                 original_image, 
                 original_text, 
                 target_text,
                 weights: LossWeights = LossWeights(clip=1.0, clip_c=0.7, mono=1.0, smoothness=1.0, color_vol=1.0, color_shift=1.0, color_eigen=1.0)):

        super().__init__()
        self.weights = weights
        self.clip_loss = CLIPLoss(original_image, original_text, target_text, content_weight=weights.clip_c)
        
        lab_original = rgb_to_lab(original_image)
        
        # 归一化 LAB 到 [0, 1] 左右的尺度，方便计算协方差
        # L: 0~100 -> 0~1
        # a, b: -128~127 -> -1~1
        lab_original[:, 0, :, :] = lab_original[:, 0, :, :] / 100.0
        lab_original[:, 1, :, :] = lab_original[:, 1, :, :] / 128.0
        lab_original[:, 2, :, :] = lab_original[:, 2, :, :] / 128.0

        b, c, h, w = lab_original.shape
        pixels = lab_original.view(b, c, -1)

        # 1. 保存原图均值 (用于 Color Shift Loss 和 协方差中心化)
        self.original_mean = pixels.mean(dim=2, keepdim=True)

        # 2. 计算并保存原图的最小特征值 (用于 Color Eigen Loss)
        # 中心化
        centered = pixels - self.original_mean
        # 计算协方差: [B, 3, 3]
        cov = torch.bmm(centered, centered.transpose(1, 2)) / (h * w - 1)
        # 防 NaN 扰动
        cov = cov + torch.eye(3, device=original_image.device).unsqueeze(0) * 1e-6
        # 原图色彩空间体积
        self.original_vol = torch.det(cov).abs()
        # 计算特征值 (实对称矩阵使用 eigvalsh 更快更稳)
        # result shape: [B, 3], 默认升序排列
        eigenvalues = torch.linalg.eigvalsh(cov)
        
        # 保存最小特征值 [B] (只取第一个，因为是升序)
        # .detach() 很重要，因为原图的特征值是常数目标，不需要梯度
        self.original_min_eigen = eigenvalues[:, 0].detach()

    def forward(self, output_image: Tensor, output_lut):
        clip_loss = self.clip_loss(output_image)
        loss_mono, loss_smooth = monotonicity_loss(output_lut)
        
        # Reuse LAB calculation for efficiency in forward pass
        lab_output = rgb_to_lab(output_image)

        lab_output[:, 0, :, :] = lab_output[:, 0, :, :] / 100.0
        lab_output[:, 1, :, :] = lab_output[:, 1, :, :] / 128.0
        lab_output[:, 2, :, :] = lab_output[:, 2, :, :] / 128.0

        b, c, h, w = lab_output.shape
        pixels = lab_output.view(b, c, -1)

        output_mean = pixels.mean(dim=2, keepdim=True)

        # 中心化
        centered = pixels - output_mean
        # 计算协方差: [B, 3, 3]
        cov = torch.bmm(centered, centered.transpose(1, 2)) / (h * w - 1)
        # 防 NaN 扰动
        cov = cov + torch.eye(3, device=output_image.device).unsqueeze(0) * 1e-6
        # 色彩空间体积
        output_vol = torch.det(cov).abs()
        # 计算特征值 (实对称矩阵使用 eigvalsh 更快更稳)
        eigenvalues = torch.linalg.eigvalsh(cov)
        
        # 保存最小特征值 [B]
        # 注意: 这里不能 detach，因为我们需要梯度回传
        output_min_eigen = eigenvalues[:, 0]

        # Loss Calculation
        # Vol Loss
        vol_loss = torch.relu(torch.log(self.original_vol + 1e-8) - torch.log(output_vol + 1e-8)).mean()

        # Shift Loss (using inline calculation for efficiency)
        shift_val = torch.norm(output_mean - self.original_mean.to(output_image.device), p=2, dim=1).mean()
        shift_loss = shift_val # Actually they are the same in this implementation

        # Eigen Loss
        eigen_loss = torch.relu(self.original_min_eigen.to(output_image.device) - output_min_eigen).mean()

        # Calculate weighted losses
        w_clip_loss = clip_loss * self.weights.clip
        w_mono_loss = loss_mono * self.weights.mono
        w_smooth_loss = loss_smooth * self.weights.smoothness
        w_vol_loss = vol_loss * self.weights.color_vol
        w_shift_loss = shift_loss * self.weights.color_shift
        w_eigen_loss = eigen_loss * self.weights.color_eigen

        total_loss = (
            w_clip_loss +
            w_mono_loss +
            w_smooth_loss +
            w_vol_loss +
            w_shift_loss +
            w_eigen_loss
        )
        
        loss_info = {
            "clip_loss": clip_loss,
            "mono_loss": loss_mono,
            "smooth_loss": loss_smooth,
            "vol_loss": vol_loss,
            "shift_loss": shift_loss,
            "eigen_loss": eigen_loss,
            "w_clip_loss": w_clip_loss,
            "w_mono_loss": w_mono_loss,
            "w_smooth_loss": w_smooth_loss,
            "w_vol_loss": w_vol_loss,
            "w_shift_loss": w_shift_loss,
            "w_eigen_loss": w_eigen_loss,
            "vol": output_vol.mean(), # Mean over batch if batch > 1
            "min_eigen": output_min_eigen.mean(),
            "shift": shift_val
        }
        
        return total_loss, loss_info 

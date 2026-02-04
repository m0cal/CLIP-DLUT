import torch
import torch.optim as optim
from torch import Tensor
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as F
from modules.clip_loss import CLIPLoss
from modules.dynamics_applier import DynamicsApplier
from modules.gradient_predictor import GradientPredictor
from modules.lut_applier import LUTApplier


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

    return loss_mono + loss_smooth

def run(image: Tensor, 
        target_text, 
        original_text="一张自然的原相机直出图", 
        epsilon = 0.001,
        lr=1e-4,
        iteration=10):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_epsilon = epsilon
    predictor = GradientPredictor(frozen=False).to(device)
    # 关闭噪声，避免随机游走导致 LUT 崩塌
    dynamics = DynamicsApplier(epsilon=dynamics_epsilon, use_noise=False).to(device)
    clip_loss = CLIPLoss(image, 
                         original_text=original_text, 
                         target_text=target_text,
                         frozen=True,
                         content_weight=0.7).to(device)
    lut_applier = LUTApplier(33).to(device)
    optimizer = optim.AdamW(predictor.parameters(), lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iteration, eta_min=1e-6) # 添加学习率调度器

    # 固定的 Identity LUT，作为锚点
    identity_lut = create_identity_lut().to(image.device)
    # 初始化当前 LUT
    current_lut = identity_lut.clone()
    
    # 预计算原图的色彩体积
    original_vol = get_color_volume(image).detach()

    for step in range(iteration):
        updated_lut = dynamics(current_lut, predictor)
        stylized_image = lut_applier(image, updated_lut)
        stylized_image = torch.clamp(stylized_image, 0, 1)  # 确保图像值在 [0, 1] 范围内

        mono_loss = monotonicity_loss(updated_lut)

        # 分离计算，避免 print 时重复 forward
        total_clip_loss = clip_loss(stylized_image)

        # 计算色彩体积损失：仅防止体积坍缩 (Hinge Loss)
        current_vol = get_color_volume(stylized_image)
        
        # 我们只惩罚体积小于原图 99% 的情况，允许由于风格化导致的体积变化（甚至增加）
        # 这样既防止了纯色化 (Vol -> 0)，又给了 CLIP 足够的优化空间
        target_vol = original_vol
        loss_volume = torch.relu(target_vol - current_vol).mean() * 1000000000.0

        loss = total_clip_loss + mono_loss + loss_volume

        optimizer.zero_grad()
        loss.backward()
        print("-" * 30)
        has_grad = False
        for name, param in predictor.named_parameters():
            if param.grad is not None:
                # .abs().mean() 可以直观看到梯度强度，避免被正负值抵消
                print(f"层: {name} | 梯度均值: {param.grad.abs().mean().item():.8f}")
                has_grad = True
            else:
                print(f"层: {name} | 警告: 没有梯度 (None)")

        if not has_grad:
            print("结果：模型完全没有收到梯度信号！")
        print("-" * 30)
        optimizer.step()
        scheduler.step() # 更新学习率

        current_lut = updated_lut.detach()
        
        print(f"Step {step}, Loss: {loss.item():.6f} (CLIP: {total_clip_loss.item():.6f}, Mono: {mono_loss.item():.6f}, Vol: {loss_volume.item():.6f})")
        
        # 每 10 步保存一次，减少 I/O
        if step % 10 == 0:
             result_img = tensor_to_image(stylized_image)
             result_img.save(f"output/output_{step}.png")


def create_identity_lut(grid_size=33):
    # 生成从 0 到 1 的等差数列
    steps = torch.linspace(0, 1, grid_size)
    
    # 创建 3D 坐标网格 (注意 indexing='ij' 对应 R, G, B)
    # meshgrid 会生成三个形状为 (33, 33, 33) 的张量
    grid_b, grid_g, grid_r = torch.meshgrid(steps, steps, steps, indexing='ij')
    
    # 拼接成 (3, 33, 33, 33) 并增加 batch 维度
    lut = torch.stack([grid_r, grid_g, grid_b], dim=0).unsqueeze(0)
    
    return lut

def image_to_tensor(image_path, downscale_to=336):
    # 1. 加载图片
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # 2. 计算缩放比例，保持长宽比
    scale = downscale_to / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 3. 等比例缩放图片
    img = F.resize(img, [new_h, new_w], interpolation=T.InterpolationMode.BICUBIC, antialias=True)

    # 4. 计算需要填充的黑边大小 (Padding)
    # 填充量 = (目标尺寸 - 当前尺寸) // 2
    pad_left = (downscale_to - new_w) // 2
    pad_top = (downscale_to - new_h) // 2
    pad_right = downscale_to - new_w - pad_left
    pad_bottom = downscale_to - new_h - pad_top

    # 5. 填充黑边 (left, top, right, bottom)
    img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0, padding_mode='constant')

    # 6. 转为 Tensor 并增加 Batch 维度
    # ToTensor 会自动将像素值归一化到 [0, 1]
    img_tensor = F.to_tensor(img).unsqueeze(0)

    return img_tensor

# 使用示例
# img_tensor = image_to_tensor("test.jpg")

def tensor_to_image(tensor):
    # 移除 Batch 维度，并将设备转回 CPU
    tensor = tensor.squeeze(0).cpu()
    # 这里的 tensor 应该是 [0, 1] 范围
    img = T.ToPILImage()(tensor)
    return img

# 保存结果
# result_img = tensor_to_image(stylized_image)
# result_img.save("output.png")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = image_to_tensor("shw.jpg", 336).to(device)
    # 增加迭代次数以允许更细致的收敛
    run(img, "一条蓝色调的巷子", original_text="一条巷子", lr=2e-4, epsilon=2e-3, iteration=1000)


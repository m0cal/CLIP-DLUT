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
from modules.loss import AllLoss, LossWeights

def run(image: Tensor, 
        target_text, 
        original_text="一张自然的原相机直出图", 
        epsilon = 0.001,
        lr=1e-4,
        iteration=10,
        progress_callback=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_epsilon = epsilon
    predictor = GradientPredictor(frozen=False).to(device)
    # 关闭噪声，避免随机游走导致 LUT 崩塌
    dynamics = DynamicsApplier(epsilon=dynamics_epsilon, use_noise=False).to(device)
    loss_weights = LossWeights(clip=1.0,
                               clip_c=0.7,
                               mono=1.0,
                               smoothness=1.0,
                               color_vol=1.0,
                               color_shift=20.0,
                               color_eigen=10000.0)
    all_loss = AllLoss(image, original_text, target_text, loss_weights)
    lut_applier = LUTApplier(33).to(device)
    optimizer = optim.AdamW(predictor.parameters(), lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iteration, eta_min=1e-6) # 添加学习率调度器

    # 固定的 Identity LUT，作为锚点
    identity_lut = create_identity_lut().to(image.device)
    # 初始化当前 LUT
    current_lut = identity_lut.clone()

    for step in range(iteration):
        updated_lut = dynamics(current_lut, predictor)
        stylized_image = lut_applier(image, updated_lut)
        stylized_image = torch.clamp(stylized_image, 0, 1)  # 确保图像值在 [0, 1] 范围内

        loss, loss_info = all_loss(stylized_image, updated_lut)

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
        
        print(f"Step {step}, Total Loss: {loss.item():.6f}")
        print(f"  CLIP Loss: {loss_info['w_clip_loss'].item():.6f} (Raw: {loss_info['clip_loss'].item():.6f})")
        print(f"  Mono Loss: {loss_info['w_mono_loss'].item():.6f} (Raw: {loss_info['mono_loss'].item():.6f}) | Smooth Loss: {loss_info['w_smooth_loss'].item():.6f} (Raw: {loss_info['smooth_loss'].item():.6f})")
        print(f"  Vol: {loss_info['vol'].item():.6f} (Loss: {loss_info['w_vol_loss'].item():.6f})")
        print(f"  Min Eigen: {loss_info['min_eigen'].item():.6f} (Loss: {loss_info['w_eigen_loss'].item():.6f})")
        if progress_callback:
            progress_callback(step, iteration, loss.item(), loss_info, stylized_image, updated_lut)

        # 移除内部的 print 和 save，改由 callback 或外部处理
        # if step % 10 == 0:
        #      result_img = tensor_to_image(stylized_image)
        #      result_img.save(f"output/output_{step}.png")
        
    return stylized_image, updated_lut


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
    img = image_to_tensor("test.jpg", 336).to(device)
    # 增加迭代次数以允许更细致的收敛
    run(img, "富有科技感的夜晚大楼", original_text="夜晚的大楼", lr=2e-4, epsilon=2e-3, iteration=1000)
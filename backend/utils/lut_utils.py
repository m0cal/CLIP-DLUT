import torch
import numpy as np
from PIL import Image
import io

def tensor_to_cube(lut_tensor: torch.Tensor, title="CLIP_DLUT") -> str:
    """
    Convert (1, 3, N, N, N) tensor to .cube string.
    Assumption: Tensor is (Batch, C, B, G, R).
    """
    if lut_tensor.dim() == 5:
        lut_tensor = lut_tensor.squeeze(0) # (3, N, N, N)
    
    C, D, H, W = lut_tensor.shape
    N = D
    
    lines = []
    lines.append(f"TITLE \"{title}\"")
    lines.append(f"LUT_3D_SIZE {N}")
    lines.append("DOMAIN_MIN 0.0 0.0 0.0")
    lines.append("DOMAIN_MAX 1.0 1.0 1.0")
    
    # Permute to (B, G, R, 3) to strictly match standard loop order:
    # Outer: B, Middle: G, Inner: R
    # Since our tensor dims are 1:B, 2:G, 3:R
    lut_permuted = lut_tensor.permute(1, 2, 3, 0) 
    
    flat_lut = lut_permuted.reshape(-1, 3).detach().cpu().numpy()
    
    # Using join is faster than printing line by line
    data_lines = [f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}" for rgb in flat_lut]
    lines.extend(data_lines)
        
    return "\n".join(lines)

def tensor_to_png_strip(lut_tensor: torch.Tensor) -> Image.Image:
    """
    Convert (1, 3, B, G, R) tensor to standard identity-style PNG strip.
    Result Image: Height=N, Width=N*N.
    Slices along Blue axis (dim 1) are concatenated horizontally.
    Each slice has Height=G (dim 2), Width=R (dim 3).
    """
    if lut_tensor.dim() == 5:
        lut_tensor = lut_tensor.squeeze(0) # (3, B, G, R)

    # Unbind along B axis -> List of (3, G, R) slices
    slices = torch.unbind(lut_tensor, dim=1) 
    
    # Concatenate along R axis (Width)
    # (3, G, R) -> (3, G, R*Steps)
    strip_tensor = torch.cat(slices, dim=2) 
    
    strip_tensor = torch.clamp(strip_tensor, 0, 1)
    
    # Convert to HWC for PIL
    # (3, H, W) -> (H, W, 3)
    img_np = (strip_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    return img

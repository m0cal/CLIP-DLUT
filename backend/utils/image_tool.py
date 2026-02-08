import io
import base64
from PIL import Image
from PIL import ImageOps as IOP
from torch import Tensor
import torchvision.transforms as T

class ImageTool:
    @staticmethod
    def base64_to_pil(v: str) -> Image.Image:
        try:
            if "," in v:
                v = v.split(",")[1]

            img_data = base64.b64decode(v)

            img = Image.open(io.BytesIO(img_data))

            img.load()

            return img
        except Exception:
            raise ValueError("Unable to decode image, seems that your image string is not in base64 format.")

    @staticmethod
    def pil_to_base64(v: Image.Image, fmt: str = "png", with_prefix: bool = True) -> str:
        
        img_data = io.BytesIO()

        v.save(img_data, format=fmt)

        img_base64 = base64.b64encode(img_data.getvalue()).decode()

        if with_prefix:
            return f"data:image/{fmt};base64,{img_base64}"
        return img_base64 

    @staticmethod
    def downscale_pil(v: Image.Image, downscale_to: int = 336) -> Image.Image:
        img = v.convert("RGB")
        w, h = img.size

        scale = downscale_to / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize(size=[new_w, new_h], resample=Image.Resampling.BICUBIC)

        img = IOP.pad(image=img, size=(downscale_to, downscale_to), color="#000000")

        return img

    @staticmethod
    def pil_to_tensor(v: Image.Image) -> Tensor:
        try:
            img_tensor = T.PILToTensor()(v)
            return img_tensor
        except Exception:
            raise ValueError("Unable to transform image to tensor.")
    
    @staticmethod
    def tensor_to_pil(v: Tensor) -> Image.Image:
        try:
            pil_image = T.ToPILImage()(v)
            return pil_image
        except Exception:
            raise ValueError("Unable to transform tensor to PIL image")

    # TODO: 别忘了归一化！
    
    @staticmethod
    def normalize_tensor(v: Tensor) -> Tensor:
        mean = Tensor([0.48145466, 0.4578275, 0.40821073], device=v.device)
        std = Tensor([0.26862954, 0.26130258, 0.27577711], device=v.device)
    
    # 适配 [C, H, W] 或 [B, C, H, W]
        if v.ndim == 3:
        # [C, 1, 1] 方便广播
            m, s = mean.view(3, 1, 1), std.view(3, 1, 1)
        else:
        # [1, C, 1, 1]
            m, s = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
        
        return (v - m) / s


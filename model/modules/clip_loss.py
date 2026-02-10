import os
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from huggingface_hub import snapshot_download

from .image_preprocesser import DifferentiableChineseCLIPProcessor

MODEL_ID = "OFA-Sys/chinese-clip-vit-large-patch14-336px"


def _get_hf_cache_dir() -> str | None:
    return os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")

def _get_hf_endpoint() -> str | None:
    return os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_BASE_URL")

def _resolve_hf_endpoint() -> str | None:
    endpoint = _get_hf_endpoint()
    if endpoint:
        return endpoint
    # If huggingface.co is not reachable, fall back to the mirror automatically.
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
        return None
    except Exception:
        return "https://hf-mirror.com"

class CLIPLoss(nn.Module):
    def __init__(
        self, 
        original_image: Tensor, 
        original_text="一张相机直出的相片", 
        target_text="一张风格化的相片",
        frozen=True,
        image_size=336,  # CLIP 模型的输入尺寸
        content_weight=0.5
    ):
        super().__init__()
        self.frozen = frozen
        self.image_size = image_size
        self.content_weight = content_weight

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # 加载模型（先确保本地缓存可用，再用本地路径加载，避免再触网）
        cache_dir = _get_hf_cache_dir()
        snapshot_path = None
        endpoint = _resolve_hf_endpoint()
        for attempt in range(3):
            try:
                if endpoint:
                    print(f"[CLIPLoss] Using HF endpoint: {endpoint}")
                snapshot_path = snapshot_download(
                    repo_id=MODEL_ID,
                    cache_dir=cache_dir,
                    endpoint=endpoint,
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[CLIPLoss] Snapshot download failed ({e}). Retry {attempt + 1}/3...")
                time.sleep(2)

        # 使用本地快照路径加载，避免继续请求 huggingface.co
        self.model = ChineseCLIPModel.from_pretrained(
            snapshot_path,
            local_files_only=True,
        ).to(device)
        self.text_processor = ChineseCLIPProcessor.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )

        # 冻结模型参数
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        self.image_processor = DifferentiableChineseCLIPProcessor().to(self.device)

        # 计算 original_text_features
        original_text_input = self.text_processor(
            text=original_text, 
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            self.original_text_features = self.model.get_text_features(**original_text_input).pooler_output
            self.original_text_features = self.original_text_features / (self.original_text_features.norm(
                p=2, dim=-1, keepdim=True
            ) + 1e-8)

            # 计算 target_text_features
            target_text_input = self.text_processor(
                text=target_text, 
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            self.target_text_features = self.model.get_text_features(**target_text_input).pooler_output
            # 同样先归一化，保证在同一尺度下计算方向
            self.target_text_features = self.target_text_features / (self.target_text_features.norm(
                p=2, dim=-1, keepdim=True
            ) + 1e-8)
            
            # 计算方向性风格特征
            self.style_features = self.target_text_features - self.original_text_features
            self.style_features = self.style_features / (self.style_features.norm(
                p=2, dim=-1, keepdim=True
            ) + 1e-8)

            # 计算原始图像特征
            original_image_input = self.image_processor(original_image)
            self.original_image_features = self.model.get_image_features(original_image_input).pooler_output
            # 归一化原始图像特征，与文本对齐
            self.original_image_features = self.original_image_features / (self.original_image_features.norm(
                p=2, dim=-1, keepdim=True
            ) + 1e-8)

    def forward(self, edited_image: Tensor) -> Tensor:
        """
        计算 CLIP 方向性损失
        
        :param edited_image: 编辑后的图像，取值范围 [0, 1]
        :type edited_image: Tensor
        :return: CLIP 损失值
        :rtype: Tensor
        """
        # 预处理图像（梯度可回传）
        edited_image_input = self.image_processor(edited_image)
        
        # 提取图像特征
        edited_image_features = self.model.get_image_features(edited_image_input).pooler_output
        edited_image_features = edited_image_features / (edited_image_features.norm(
            p=2, dim=-1, keepdim=True
        ) + 1e-8)
        
        # 计算编辑方向，归一化
        edit_direction = edited_image_features - self.original_image_features
        edit_direction = edit_direction / (edit_direction.norm(
            p=2, dim=-1, keepdim=True
        ) + 1e-8)
        # 计算方向性损失（余弦相似度）
        # 我们希望编辑方向与风格方向一致
        directional_loss = 1.0 - torch.cosine_similarity(
            edit_direction, 
            self.style_features, 
            dim=-1
        ).mean()

        content_loss = 1.0 - torch.cosine_similarity(
            edited_image_features,
            self.target_text_features,
            dim=-1
        ).mean()
        
        return directional_loss*10.0 + self.content_weight * content_loss

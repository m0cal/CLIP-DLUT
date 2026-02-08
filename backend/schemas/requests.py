from pydantic import BaseModel, Field, field_validator
from PIL import Image
import base64
import io
from uuid import UUID
from .tasks import LUTFormat
from ..utils.image_tool  import ImageTool


class RetouchingRequest(BaseModel):
    image: str = Field(...,description="需要调色的图片，Base64 格式，最好是 336*336 大。")
    target_prompt: str = Field(..., description="目标风格描述，需要带上图片内容，比如“一条赛博朋克色调的街道”。")
    original_prompt: str | None = Field(default="一张自然色调的图片", description="图片内容描述。")
    iteration: int | None = Field(default=1000, ge=1, le=5000, description="迭代处理次数。")


class QueryTaskRequest(BaseModel):
    task_id: UUID = Field(..., description="任务唯一标识。")
    lut_format: LUTFormat | None = Field(default=None, description="返回 LUT 格式，可选'cube'或'png'，建议使用 png。不传则不返回。")
    include_image: bool = Field(default=False, description="是否返回缩略图。")

class StopTaskRequest(BaseModel):
    task_id: UUID = Field(...,description="任务唯一标识。")


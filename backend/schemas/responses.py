from PIL import Image
import base64
import io
from uuid import UUID
from .tasks import TaskStatus

from pydantic import BaseModel, Field

class RetouchingResponse(BaseModel):
    task_id: UUID = Field(..., description="任务唯一标识，后续查询和删除需要用到")
    status: TaskStatus = Field(description="任务状态，可能为 pending, processing, finished")
    current_iteration: int = Field(default=0, description="当前迭代次数")
    overall_iteration: int = Field(default=1000, description="总迭代次数")
    lut: str | None = Field(default=None, description="根据需求，返回对应形式的 LUT，如果是 PNG，返回 Base64 编码的 PNG")
    image: str | None = Field(default=None, description="根据需求，返回缩略图的 Base64 编码。")
    

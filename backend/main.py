from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import torch

from .schemas.requests import RetouchingRequest, QueryTaskRequest, StopTaskRequest
from .schemas.responses import RetouchingResponse
from .schemas.tasks import TaskStatus
from .core.celery_app import celery_app
from .tasks.processing import process_image_task
from .core.config import settings
from .utils.image_tool import ImageTool
from .utils.lut_utils import tensor_to_cube, tensor_to_png_strip
from celery.result import AsyncResult
import json
import redis
from PIL import Image
import base64

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to CLIP-DLUT Backend"}

@app.post("/retouch", response_model=RetouchingResponse)
async def create_retouch_task(request: RetouchingRequest):
    task_id = str(uuid.uuid4())
    iteration = max(1, min(5000, int(request.iteration)))
    
    # Send to Celery
    # We pass the image as base64 to avoid passing large data if possible, 
    # but here we already have it in request.image (which is Image.Image via validator).
    # Celery needs simple args (json serializable).
    # So we need to convert Image back to base64 or save to temp file and pass path.
    # Passing large Base64 to Redis is not ideal but easiest for now.
    # Optimally: Save to disk, pass path.
    
    # Validate Base64
    try:
        # Check if it's a valid image string
        ImageTool.base64_to_pil(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    
    task = process_image_task.apply_async(
        args=[
            request.image,
            request.target_prompt,
            request.original_prompt,
            iteration,
            task_id
        ],
        task_id=task_id
    )
    
    return RetouchingResponse(
        task_id=uuid.UUID(task_id),
        status=TaskStatus.PENDING,
        current_iteration=0,
        overall_iteration=iteration,
        lut=None,
        image=None # Not returning full image in immediate response
    )

@app.post("/query_task", response_model=RetouchingResponse)
async def query_task(request: QueryTaskRequest):
    task_result = AsyncResult(str(request.task_id), app=celery_app)
    
    status = TaskStatus.PENDING
    current_iter = 0
    overall_iter = 1000
    lut_data = None
    image_data = None
    error_message = None
    
    celery_status = task_result.status
    # Directly read backend meta; Redis fallback handles cases where celery_status stays PENDING.
    task_meta = task_result.backend.get_task_meta(task_result.id) if task_result.backend else {}

    def _normalize_meta_object(meta_obj):
        if isinstance(meta_obj, dict):
            return meta_obj
        if isinstance(meta_obj, (bytes, str)):
            try:
                parsed = json.loads(meta_obj)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _apply_progress_meta(meta: dict) -> bool:
        nonlocal status, current_iter, overall_iter, image_data
        meta = _normalize_meta_object(meta)
        if not meta:
            return False

        meta_status = meta.get("status")
        meta_status_norm = meta_status.lower() if isinstance(meta_status, str) else None
        meta_result = meta.get("result")
        meta_payload = meta_result if isinstance(meta_result, dict) else meta
        if not isinstance(meta_payload, dict):
            return False

        has_progress = ("current_iteration" in meta_payload) or ("overall_iteration" in meta_payload)
        if not has_progress:
            return False

        meta_current = int(meta_payload.get("current_iteration") or 0)
        meta_overall = int(meta_payload.get("overall_iteration") or overall_iter)

        # Some backends may temporarily expose "pending" while result already has progress.
        if meta_status_norm in ("processing", "started") or meta_current > 0:
            status = TaskStatus.PROCESSING

        if meta_current > current_iter:
            current_iter = meta_current
        if meta_overall > 0:
            overall_iter = meta_overall

        if request.include_image and meta_payload.get("preview"):
            image_data = meta_payload.get("preview")
        return True

    live_task = celery_status in ('PENDING', 'STARTED', 'PROCESSING')
    if live_task:
        _apply_progress_meta(task_meta)

        # Redis raw meta fallback: robust for Celery states that lag behind actual progress.
        if current_iter == 0:
            try:
                r = redis.Redis(host="localhost", port=6379, db=0)
                raw = r.get(f"celery-task-meta-{task_result.id}")
                if raw:
                    _apply_progress_meta(raw)
            except Exception:
                pass

        if status == TaskStatus.PENDING and celery_status in ('STARTED', 'PROCESSING'):
            status = TaskStatus.PROCESSING

        if status == TaskStatus.PROCESSING and isinstance(task_result.info, dict):
            info_current = int(task_result.info.get("current_iteration") or 0)
            info_overall = int(task_result.info.get("overall_iteration") or overall_iter)
            if info_current > current_iter:
                current_iter = info_current
            if info_overall > 0:
                overall_iter = info_overall
            if request.include_image and task_result.info.get("preview"):
                image_data = task_result.info.get("preview")

    if celery_status == 'SUCCESS':
        status = TaskStatus.FINISHED
        result = task_result.result
        # The worker returns specific dict
        current_iter = result.get("iteration", overall_iter) if isinstance(result, dict) else overall_iter
        overall_iter = result.get("iteration", overall_iter) if isinstance(result, dict) else overall_iter
        
        # Load result image to return base64 if requested
        if request.include_image and result and "final_image_path" in result:
             try:
                 with Image.open(result["final_image_path"]) as img:
                     image_data = ImageTool.pil_to_base64(img)
             except Exception:
                 pass
                 
        # If LUT requested
        if request.lut_format and result and "lut_path" in result:
             try:
                 lut_tensor = torch.load(result["lut_path"], map_location="cpu")
                 if request.lut_format == 'cube':
                     lut_str = tensor_to_cube(lut_tensor)
                     # Return as base64 encoded string to avoid encoding issues with JSON
                     lut_data = base64.b64encode(lut_str.encode('utf-8')).decode('utf-8')
                 elif request.lut_format == 'png': # or png
                     lut_img = tensor_to_png_strip(lut_tensor)
                     lut_data = ImageTool.pil_to_base64(lut_img)
             except Exception as e:
                 print(f"Error loading LUT: {e}")
                 pass
        
    elif celery_status == 'FAILURE':
        status = TaskStatus.FAILED
        if isinstance(task_result.result, Exception):
            error_message = str(task_result.result)
        elif task_result.result is not None:
            error_message = str(task_result.result)
        elif isinstance(task_meta, dict) and task_meta.get("traceback"):
            traceback_lines = str(task_meta.get("traceback")).strip().splitlines()
            error_message = traceback_lines[-1] if traceback_lines else "Task failed"
    elif celery_status == 'REVOKED':
        status = TaskStatus.STOPPED
    elif celery_status == 'PENDING' and status != TaskStatus.PROCESSING:
        status = TaskStatus.PENDING

    return RetouchingResponse(
        task_id=request.task_id,
        status=status,
        current_iteration=current_iter,
        overall_iteration=overall_iter,
        image=image_data,
        lut=lut_data,
        error=error_message,
    )

@app.post("/stop_task")
async def stop_task(request: StopTaskRequest):
    celery_app.control.revoke(str(request.task_id), terminate=True)
    return {"status": "ok", "task_id": request.task_id}

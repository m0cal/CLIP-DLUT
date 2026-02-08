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
            request.iteration,
            task_id
        ],
        task_id=task_id
    )
    
    return RetouchingResponse(
        task_id=uuid.UUID(task_id),
        status=TaskStatus.PENDING,
        current_iteration=0,
        overall_iteration=request.iteration,
        lut=None,
        image=None # Not returning full image in immediate response
    )

@app.post("/query_task", response_model=RetouchingResponse)
async def query_task(request: QueryTaskRequest):
    task_result = AsyncResult(str(request.task_id), app=celery_app)
    
    status = TaskStatus.PENDING
    current_iter = 0
    lut_data = None
    image_data = None
    
    celery_status = task_result.status
    # Directly read backend meta to avoid PENDING fallback when state exists
    task_meta = task_result.backend.get_task_meta(task_result.id) if task_result.backend else {}
    meta_status = task_meta.get("status") if isinstance(task_meta, dict) else None
    meta_result = task_meta.get("result") if isinstance(task_meta, dict) else None
    meta_status_norm = meta_status.lower() if isinstance(meta_status, str) else None
    
    if meta_status_norm in ("processing", "started") and isinstance(meta_result, dict):
        status = TaskStatus.PROCESSING
        current_iter = meta_result.get("current_iteration", 0)
        if request.include_image:
            image_data = meta_result.get("preview", None)
    elif meta_status_norm is None and celery_status == 'PENDING':
        # Fallback: read raw meta from Redis directly
        try:
            r = redis.Redis(host="localhost", port=6379, db=0)
            raw = r.get(f"celery-task-meta-{task_result.id}")
            if raw:
                meta = json.loads(raw)
                meta_status = meta.get("status")
                meta_result = meta.get("result")
                meta_status_norm = meta_status.lower() if isinstance(meta_status, str) else None
                if meta_status_norm in ("processing", "started") and isinstance(meta_result, dict):
                    status = TaskStatus.PROCESSING
                    current_iter = meta_result.get("current_iteration", 0)
                    if request.include_image:
                        image_data = meta_result.get("preview", None)
        except Exception:
            pass
    elif celery_status == 'PENDING':
        status = TaskStatus.PENDING
    elif celery_status == 'STARTED' or celery_status == 'PROCESSING': # Custom state or standard STARTED
        status = TaskStatus.PROCESSING
        # Get progress info
        if task_result.info and isinstance(task_result.info, dict):
            current_iter = task_result.info.get("current_iteration", 0)
            if request.include_image:
                image_data = task_result.info.get("preview", None)
    elif celery_status == 'SUCCESS':
        status = TaskStatus.FINISHED
        result = task_result.result
        # The worker returns specific dict
        current_iter = result.get("iteration", 1000) if isinstance(result, dict) else 1000
        
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
    elif celery_status == 'REVOKED':
        status = TaskStatus.STOPPED

    return RetouchingResponse(
        task_id=request.task_id,
        status=status,
        current_iteration=current_iter,
        image=image_data if image_data else "", # Schema expects str, not None? Validator might complain if None passed for str field. Check schema default.
        lut=lut_data 
    )

@app.post("/stop_task")
async def stop_task(request: StopTaskRequest):
    celery_app.control.revoke(str(request.task_id), terminate=True)
    return {"status": "ok", "task_id": request.task_id}


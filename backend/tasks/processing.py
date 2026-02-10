import sys
import os
import time
import torch
import torchvision.transforms.functional as VF
import traceback
import gc
from celery import Task
from celery.utils.log import get_task_logger

# Add the project root and model directory to sys.path to allow imports from model/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
model_dir = os.path.join(project_root, "model")

if project_root not in sys.path:
    sys.path.append(project_root)
if model_dir not in sys.path:
    sys.path.append(model_dir)

from backend.core.celery_app import celery_app
from backend.core.config import settings
from backend.utils.image_tool import ImageTool
from backend.schemas.tasks import TaskStatus, LUTFormat

# Import model components
# Since we added model_dir to sys.path, we can import run directly or modules directly
# model/run.py acts as a script but has functions. 
# Its imports form modules.* will work because we added model_dir to sys.path
try:
    from model.run import run as run_model, image_to_tensor, tensor_to_image
    from modules.lut_applier import LUTApplier
except ImportError:
    # Log error if import fails, likely path issue
    get_task_logger(__name__).error("Failed to import model.run. Make sure 'model' directory is in path.")
    raise

logger = get_task_logger(__name__)

class PredictTask(Task):
    """
    Abstact CLI task that handles errors and logging
    """
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)

@celery_app.task(bind=True, base=PredictTask)
def process_image_task(self, 
                       image_base64: str, 
                       target_prompt: str, 
                       original_prompt: str, 
                       iteration: int,
                       task_id: str):
                       
    try:
        iteration = max(1, min(5000, int(iteration)))
        logger.info(f"Task {task_id}: configured iteration={iteration}")

        # 1. Setup paths
        work_dir = os.path.join(settings.RESULT_DIR, task_id)
        os.makedirs(work_dir, exist_ok=True)
        
        # 2. Decode Image
        image_pil = ImageTool.base64_to_pil(image_base64)
        input_path = os.path.join(work_dir, "input.png")
        image_pil.save(input_path) # Backup input
        
        # 3. Prepare Model Inputs
        # image_to_tensor resizes and pads to square (default 336)
        # We might need to handle arbitrary sizes or trust the utility
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = image_to_tensor(input_path, 336).to(device)
        
        # 4. Define Progress Callback
        # Push an initial status so frontend doesn't stay on an empty state.
        self.update_state(
            state='PROCESSING',
            meta={
                "current_iteration": 0,
                "overall_iteration": iteration,
                "phase": "initializing"
            }
        )

        last_preview_ts = 0.0

        def _should_emit_preview(current_iteration: int, total: int) -> bool:
            nonlocal last_preview_ts
            now = time.time()
            if current_iteration <= 1 or current_iteration >= total:
                last_preview_ts = now
                return True
            if now - last_preview_ts >= 2.0:
                last_preview_ts = now
                return True
            return False

        def callback(step, total, loss_val, loss_info, stylized_tensor, updated_lut):
            # Report human-friendly progress (1..total), not 0-based step.
            current_iteration = step + 1

            # Update iteration state frequently for smooth progress display.
            # Only encode preview periodically to reduce CPU and memory overhead.
            emit_preview = _should_emit_preview(current_iteration, total)
            meta = {
                "current_iteration": current_iteration,
                "overall_iteration": total,
                "loss": float(loss_val),
            }

            if emit_preview:
                current_img = tensor_to_image(stylized_tensor)
                current_img.thumbnail((256, 256))
                meta["preview"] = ImageTool.pil_to_base64(current_img)

            self.update_state(state='PROCESSING', meta=meta)


        # 5. Run Model
        logger.info(f"Starting model run for task {task_id}")
        final_tensor, final_lut = run_model(
            image=img_tensor,
            target_text=target_prompt,
            original_text=original_prompt,
            iteration=iteration,
            progress_callback=callback
        )
        
        # 6. Save Results
        # Apply final LUT to the original-resolution image so output keeps original size
        with torch.no_grad():
            full_img_tensor = VF.to_tensor(image_pil).unsqueeze(0).to(device)
            full_lut_applier = LUTApplier(final_lut.shape[-1]).to(device)
            full_img_tensor = full_lut_applier(full_img_tensor, final_lut)
            full_img_tensor = torch.clamp(full_img_tensor, 0, 1)

        final_img = tensor_to_image(full_img_tensor)
        final_img_path = os.path.join(work_dir, "final_result.png")
        final_img.save(final_img_path)
        
        # Save LUT (Converting Tensor LUT to CUBE or helper format)
        # The run.py returns a Tensor LUT. 
        # We need a way to save it as .cube or .png. 
        # For now, let's just torch.save it, or if there is a util in modules...
        # 'modules/lut_applier.py' might have it, but standard is .cube
        # Let's just save the tensor for now
        lut_path = os.path.join(work_dir, "lut.pt")
        torch.save(final_lut, lut_path)
        
        return {
            "status": TaskStatus.FINISHED,
            "final_image_path": final_img_path,
            "lut_path": lut_path,
            "iteration": iteration
        }
        
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
    finally:
        # Force garbage collection and empty CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Task {task_id}: CUDA cache emptied.")

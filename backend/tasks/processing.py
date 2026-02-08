import sys
import os
import torch
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
        def callback(step, total, loss_val, loss_info, stylized_tensor, updated_lut):
            # Only update state every N steps to save Redis overhead
            if step % 10 == 0 or step == total - 1:
                # Convert current result to base64 for preview
                # Note: This might be slow if image is larg. 
                # tensor_to_image returns CPU PIL image
                current_img = tensor_to_image(stylized_tensor)
                
                # Check if we need to save intermediate result for history
                # current_img.save(os.path.join(work_dir, f"preview_{step}.png"))
                
                # Create thumbnail for frontend
                current_img.thumbnail((256, 256))
                preview_b64 = ImageTool.pil_to_base64(current_img)
                
                # IMPORTANT: Use 'STARTED' or custom state, but API checks for PROCESSING
                # We map 'PROCESSING' string to state
                self.update_state(
                    state='PROCESSING', 
                    meta={
                        "current_iteration": step,
                        "overall_iteration": total,
                        "loss": loss_val,
                        "preview": preview_b64
                    }
                )


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
        final_img = tensor_to_image(final_tensor)
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

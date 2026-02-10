import sys
import os
from PIL import Image, ImageFilter
import gradio as gr
import torch
import tempfile

# Get absolute path of the current directory (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get path to the 'model' directory
model_dir = os.path.join(current_dir, "model")

# Add 'model' directory to sys.path so that 'import modules...' inside run.py works
if model_dir not in sys.path:
    sys.path.append(model_dir)

# Now we can safely import from model.run
try:
    # Since we added model_dir to sys.path, we can import run directly if we want,
    # or import via model.run if the top level is also in path.
    # However, run.py does "from modules import...", which works because model_dir is in path.
    from model.run import run, image_to_tensor, tensor_to_image
except ImportError:
    # Fallback: try importing as if 'run' is a top level module because of sys.path append
    from run import run, image_to_tensor, tensor_to_image

# Import LUT utilities
from backend.utils.lut_utils import tensor_to_cube

def process_image(image_path, target_text, original_text, epsilon, lr, iteration, progress=gr.Progress()):
    if image_path is None:
        return None, None
    
    if not target_text:
        target_text = "Standard image"
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Preprocess the image
    try:
        # Load low-res for training (336 is the default downscale size from run.py)
        img_tensor = image_to_tensor(image_path, downscale_to=336).to(device)
    except Exception as e:
        raise gr.Error(f"Error loading image: {str(e)}")
    
    # 2. Define a progress callback
    def gradio_callback(step, total, loss, loss_info, current_img, current_lut):
        progress(step / total, desc=f"Step {step}/{total} | Loss: {loss:.4f}")
    
    # 3. Run the styling process
    try:
        stylized_tensor, updated_lut = run(
            image=img_tensor,
            target_text=target_text,
            original_text=original_text,
            epsilon=epsilon,
            lr=lr,
            iteration=int(iteration),
            progress_callback=gradio_callback
        )
    except Exception as e:
        raise gr.Error(f"Error during model execution: {str(e)}")
    
    # 4. Apply 3D LUT to original High-Res image using Pillow
    try:
        # Process LUT: (1, 3, 33, 33, 33) -> (Batch, Channels, Depth(B), Height(G), Width(R))
        lut = updated_lut.detach().cpu().squeeze(0) # (3, 33, 33, 33)
        lut = torch.clamp(lut, 0.0, 1.0)
        
        # Permute to (Depth, Height, Width, Channels) for flattening
        # Pillow Color3DLUT expects: R changes fastest, then G, then B.
        # Our tensor dims correspond to indexes (from run.py meshgrid): dim 1->B, dim 2->G, dim 3->R
        # So we permute to (1, 2, 3, 0) => (B_idx, G_idx, R_idx, Channels)
        lut = lut.permute(1, 2, 3, 0) # (33, 33, 33, 3)
        
        lut_values = lut.flatten().tolist()
        
        # Load original image
        original_img = Image.open(image_path).convert("RGB")
        
        # Create and apply filter
        lut_filter = ImageFilter.Color3DLUT(33, lut_values)
        result_img = original_img.filter(lut_filter)
        
        # Generate .cube file
        cube_content = tensor_to_cube(updated_lut, title="CLIP_DLUT_Result")
        
        # Create a temporary file for the .cube
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as f:
            f.write(cube_content)
            cube_file_path = f.name
        
        return result_img, cube_file_path
    except Exception as e:
        print(f"Error applying LUT: {e}")
        # Fallback to low-res result if something fails
        return tensor_to_image(stylized_tensor), None

# Create Gradio Interface
with gr.Blocks(title="CLIP-DLUT Style Transfer") as demo:
    gr.Markdown(
        """
        # CLIP-DLUT Zero-Shot Image Styling
        Apply text-driven color grading and style transfer to your images using CLIP-DLUT.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image_comp = gr.Image(type="filepath", label="Input Image", height=336)
            
            target_text_comp = gr.Textbox(
                label="Target Style Description", 
                placeholder="e.g. A cyberpunk night city with neon lights", 
                value="赛博朋克的夜晚大楼",
                info="Describe the visual style you want to achieve."
            )
            
            original_text_comp = gr.Textbox(
                label="Original Image Description (Optional)", 
                value="相机直出的风景或人物",
                info="Describe the content of the original image for better semantic preservation."
            )
            
            with gr.Accordion("Hyperparameters", open=False):
                epsilon_comp = gr.Number(label="Epsilon (Dynamics)", value=1e-4, step=1e-4)
                lr_comp = gr.Number(label="Learning Rate", value=2e-4, step=1e-5)
                iteration_comp = gr.Slider(
                    minimum=10, 
                    maximum=2000, 
                    value=1000, 
                    step=10, 
                    label="Iterations"
                )
            
            run_btn = gr.Button("Generate Style", variant="primary")
            
        with gr.Column(scale=1):
            output_image_comp = gr.Image(label="Stylized Result")
            output_cube_comp = gr.File(label="Download LUT (.cube)")

    run_btn.click(
        fn=process_image,
        inputs=[
            input_image_comp, 
            target_text_comp, 
            original_text_comp, 
            epsilon_comp, 
            lr_comp, 
            iteration_comp
        ],
        outputs=[output_image_comp, output_cube_comp]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

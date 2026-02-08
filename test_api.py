import time
import requests
import base64
import io
import sys
from PIL import Image, ImageFilter
import numpy as np

API_URL = "http://127.0.0.1:8000"

import os

def apply_cube_lut(image_path: str, lut_content: str, output_path: str):
    """
    Apply a .cube LUT to an image using Pillow's Color3DLUT.
    """
    print(f"🎨 Applying LUT to {image_path}...")
    try:
        # 1. Parse .cube content
        lines = lut_content.splitlines()
        size = 33
        table = []
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 1.0]

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("TITLE"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if line.startswith("DOMAIN_MIN"):
                domain_min = [float(x) for x in line.split()[1:]]
                continue
            if line.startswith("DOMAIN_MAX"):
                domain_max = [float(x) for x in line.split()[1:]]
                continue
            
            # Helper to parse floats safely
            parts = line.split()
            if len(parts) == 3:
                vals = [float(x) for x in parts]
                table.extend(vals)

        if len(table) != size * size * size * 3:
            print(f"⚠️ Warning: LUT table size mismatch. Expected {size**3 * 3}, got {len(table)}")

        # 2. Create Pillow LUT Filter
        # Pillow expects the table to be flat list of R, G, B values
        # The .cube standard order (B varies slowest, R varies fastest) matches Pillow's expectation
        lut_filter = ImageFilter.Color3DLUT(size, table)
        
        # 3. Apply to image
        with Image.open(image_path).convert('RGB') as img:
            # Resize for speed or consistency if needed, but LUT works on any size
            # We use original size here to verify full effect
            result = img.filter(lut_filter)
            result.save(output_path)
            print(f"✅ Saved client-side LUT applied image to: {output_path}")

    except Exception as e:
        print(f"❌ Failed to apply LUT locally: {e}")

def load_test_image_b64():
    """Load model/test.jpg for testing."""
    # Try to load existing test image
    image_path = os.path.join("model", "shw.jpg")
    if os.path.exists(image_path):
        print(f"📄 Loading test image from: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
            
    print("⚠️ Test image not found, using dummy red square.")
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

def check_health():
    """Check if the API is reachable."""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API is reachable.")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running?")
        return False

def normalize_base64(data: str) -> str:
    """Strip data URL prefix and fix missing padding for base64 strings."""
    if "," in data:
        data = data.split(",", 1)[1]
    padding = (-len(data)) % 4
    if padding:
        data += "=" * padding
    return data

def run_test():
    print(f"Testing CLIP-DLUT Backend at {API_URL}...")
    
    if not check_health():
        return

    # 1. Submit Taks
    print("\n1. Submitting Retouch Task...")
    payload = {
        "image": load_test_image_b64(),
        "target_prompt": "一条蓝色调的巷子",
        "original_prompt": "一条巷子", 
        "iteration": 1000 # Incremented iteration for better effect
    }
    
    try:
        resp = requests.post(f"{API_URL}/retouch", json=payload)
        resp.raise_for_status()
        data = resp.json()
        task_id = data['task_id']
        print(f"✅ Task submitted successfully. Task ID: {task_id}")
    except Exception as e:
        print(f"❌ Failed to submit task: {e}")
        if resp:
            print(f"Response: {resp.text}")
        return

    # 2. Poll Status
    print(f"\n2. Polling Task Status (Task ID: {task_id})...")
    
    start_time = time.time()
    while True:
        try:
            query_resp = requests.post(
                f"{API_URL}/query_task", 
                json={
                    "task_id": task_id, 
                    "include_image": True,
                    "lut_format": "cube" 
                }
            )
            query_resp.raise_for_status()
            status_data = query_resp.json()
            status = status_data['status']
            
            elapsed = time.time() - start_time
            print(f"   [{elapsed:.1f}s] Status: {status}, Iteration: {status_data.get('current_iteration')}")
            
            if status == "finished":
                print("\n✅ Task Finished!")
                
                lut_content_str = None
                
                if status_data.get('lut'):
                    print("   - LUT received")
                    try:
                        lut_b64 = normalize_base64(status_data['lut'])
                        lut_bytes = base64.b64decode(lut_b64)
                        
                        # Save as .cube
                        lut_content_str = lut_bytes.decode('utf-8')
                        with open("output.cube", "w") as f:
                            f.write(lut_content_str)
                        print("   - Saved LUT to output.cube")
                    except Exception as e:
                        print(f"   - Failed to save LUT: {e}")

                if status_data.get('image'):
                    print("   - Result image received")
                    try:
                        img_b64 = normalize_base64(status_data['image'])
                        img_bytes = base64.b64decode(img_b64)
                        with open("server_result.png", "wb") as f:
                            f.write(img_bytes)
                        print("   - Saved server result to server_result.png")
                    except Exception as e:
                        print(f"   - Failed to save Image: {e}")

                # Apply LUT locally to verify
                if lut_content_str:
                    input_image_path = os.path.join("model", "test.jpg")
                    if os.path.exists(input_image_path):
                         apply_cube_lut(input_image_path, lut_content_str, "client_lut_applied.png")
                    else:
                        print("⚠️ Skipping local LUT application: Input image model/test.jpg not found.")

                break
            
            if status == "failed":
                print("\n❌ Task Failed!")
                break
            
            if status == "stopped":
                print("\n⚠️ Task Stopped!")
                break
                
            time.sleep(2)
            
            if elapsed > 1000: # Timeout
                print("\n❌ Test Timed Out (1000s).")
                break
                
        except Exception as e:
            print(f"❌ Error polling status: {e}")
            break

if __name__ == "__main__":
    run_test()

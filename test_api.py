import time
import requests
import base64
import io
import sys
from PIL import Image

API_URL = "http://127.0.0.1:8000"

import os

def load_test_image_b64():
    """Load model/test.jpg for testing."""
    # Try to load existing test image
    image_path = os.path.join("model", "test.jpg")
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

def run_test():
    print(f"Testing CLIP-DLUT Backend at {API_URL}...")
    
    if not check_health():
        return

    # 1. Submit Taks
    print("\n1. Submitting Retouch Task...")
    payload = {
        "image": load_test_image_b64(),
        "target_prompt": "Cyberpunk style, neon lights",
        "original_prompt": "Night skyscrapers", 
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
                    "lut_format": "png" 
                }
            )
            query_resp.raise_for_status()
            status_data = query_resp.json()
            status = status_data['status']
            
            elapsed = time.time() - start_time
            print(f"   [{elapsed:.1f}s] Status: {status}, Iteration: {status_data.get('current_iteration')}")
            
            if status == "finished":
                print("\n✅ Task Finished!")
                if status_data.get('lut'):
                    print("   - LUT received (Base64 PNG)")
                    # Save LUT
                    try:
                        lut_bytes = base64.b64decode(status_data['lut'])
                        with open("output_lut.png", "wb") as f:
                            f.write(lut_bytes)
                        print("   - Saved LUT to output_lut.png")
                    except Exception as e:
                        print(f"   - Failed to save LUT: {e}")

                if status_data.get('image'):
                    print("   - Preview image received")
                    try:
                        img_bytes = base64.b64decode(status_data['image'])
                        with open("output_preview.png", "wb") as f:
                            f.write(img_bytes)
                        print("   - Saved result to output_preview.png")
                    except Exception as e:
                        print(f"   - Failed to save Image: {e}")
                break
            
            if status == "failed":
                print("\n❌ Task Failed!")
                break
            
            if status == "stopped":
                print("\n⚠️ Task Stopped!")
                break
                
            time.sleep(2)
            
            if elapsed > 600: # Timeout
                print("\n❌ Test Timed Out (60s).")
                break
                
        except Exception as e:
            print(f"❌ Error polling status: {e}")
            break

if __name__ == "__main__":
    run_test()

import os
import threading
import time
import webbrowser
import subprocess
import sys

from Enhancement import enhance_image
from Image_segmentation import run_inference
from Severity_analysis import analyze_and_export
from pathlib import Path

DRONE_INPUT = "drone_input"
ENH_OUT = "enhance_output"
MASK_OUT = "mask_output"
SEV_OUT = "severity_output"
BASE_DIR = Path(__file__).parent
WEBSITE_DIR = BASE_DIR / "Website"

os.makedirs(ENH_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)
os.makedirs(SEV_OUT, exist_ok=True)

def get_single_file(folder):
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    if len(files) == 0:
        raise FileNotFoundError(f"No image found in {folder}")
    return os.path.join(folder, files[0])

def start_llm_server():
    subprocess.run(
        [sys.executable, str(WEBSITE_DIR / "llm_server.py")],
        cwd=WEBSITE_DIR)

def start_website():
    subprocess.run(
        [sys.executable, "-m", "http.server", "8000"],
        cwd=BASE_DIR)

def main():
    print("\n=== ResQ-Vision STARTED ===")

    # -------------------------------------
    # 1. ENHANCEMENT
    # -------------------------------------
    print("\n[1] Enhancing Image...")
    raw_img = get_single_file(DRONE_INPUT)
    enhanced_path = enhance_image(raw_img)

    # -------------------------------------
    # 2. SEGMENTATION
    # -------------------------------------
    print("\n[2] Running Segmentation...")
    gray_mask_path = run_inference(
        enhanced_path,
        MASK_OUT,          
        SEV_OUT
    )

    # -------------------------------------
    # 3. SEVERITY ANALYSIS
    # -------------------------------------
    print("\n[3] Running Severity Analysis...")
    json_file, result = analyze_and_export(gray_mask_path)
    
    # -------------------------------------
    # 4. LLM SERVER
    # -------------------------------------
    print("\n[4] Starting LLM server...")
    threading.Thread(target=start_llm_server, daemon=True).start()
    time.sleep(4)

	# -------------------------------------
    # 5. WEBSITE SERVER
    # -------------------------------------
    print("\n[5] Starting Website server...")
    threading.Thread(target=start_website, daemon=True).start()
    time.sleep(2)


    print("\n[6] Opening dashboard...")
    webbrowser.open("<your dashboard url>")

    while True:
        time.sleep(1)

    print("\n=== ResQ-Vision COMPLETED SUCCESSFULLY ===\n")
    
if __name__ == "__main__":
    main()
 

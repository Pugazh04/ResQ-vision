import os
import sys
import cv2
import numpy as np

# ---------------------------
# Enhancement Function
# ---------------------------
def _enhance_opencv(img_bgr: np.ndarray, upscale: int = 2) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    img_up = cv2.resize(img_bgr, (w*upscale, h*upscale), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img_up, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    img_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(img_clahe, (0,0), sigmaX=1.0)
    sharpen = cv2.addWeighted(img_clahe, 2.0, blur, -1.0, 0)

    return sharpen

def enhance_image(in_path: str) -> str:
    out_dir= 'enhance_output'
    out_path = os.path.join(out_dir, "enhance_img.jpg")

    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    enhanced = _enhance_opencv(img, upscale=2)
    enhanced = cv2.resize(enhanced, (384, 384), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, enhanced)
    print(f"Saved Enhanced Image → {out_path}")

    return out_path

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import segmentation_models_pytorch as smp

NUM_CLASSES = 12
TEST_RES = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH  = "deeplabv3plus_mixed.pth"

COLOR_MAP = {
    0:  (0, 0, 0),        # Background - Black
    1:  (0, 0, 139),      # Water - Dark Blue
    2:  (255, 255, 255),  # B-NoDamage - White
    3:  (255, 255, 0),    # B-MedDamage - Yellow
    4:  (255, 165, 0),    # B-MajDamage - Orange
    5:  (255, 0, 0),      # B-Destruction - Red
    6:  (238, 130, 238),  # Vehicle - Violet
    7:  (128, 128, 128),  # Road-Clear - Grey
    8:  (255, 0, 255),    # Road-Blocked - Magenta
    9:  (0, 255, 0),      # Tree - Green
    10: (0, 0, 0),        # Pool - Black
    11: (0, 0, 0)         # Other - Black
}

def decode_color(mask):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cid, color in COLOR_MAP.items():
        out[mask == cid] = color
    return out

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
try:
    model.load_state_dict(state)
except:
    model.load_state_dict(state, strict=False)

model.eval()
print("DeepLabV3 model loaded successfully !!!")

normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

def run_inference(input_image_path, output_folder, output_folder1):

    img_bgr = cv2.imread(input_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = img_rgb

    img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    gray_out_path = os.path.join(output_folder, "mask_img.png")
    cv2.imwrite(gray_out_path, pred)
    print(f"Saved Grayscale Mask → {gray_out_path}")

    pred_color = decode_color(pred)
    
    overlay = (0.6 * img_resized.astype(np.float32) +
               0.4 * pred_color.astype(np.float32)).astype(np.uint8)
    color_out_path = os.path.join(output_folder1, "color_mask_img.png")
    cv2.imwrite(color_out_path, overlay)
    print(f"Saved Colorscale Mask → {color_out_path}")

    return gray_out_path



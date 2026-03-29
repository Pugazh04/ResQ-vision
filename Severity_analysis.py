import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------
# EfficientNet-B0 loader for state_dict
# -------------------------
import torch.nn as nn
from torchvision.models import efficientnet_b0

def build_effnet_b0(num_classes=5):
    model = efficientnet_b0(weights=None)
    old_conv = model.features[0][0]
    old_weight = old_conv.weight.detach().clone()
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )

    new_conv.weight.data[:, :3, :, :] = old_weight
    new_conv.weight.data[:, 3:4, :, :] = 0.0
    model.features[0][0] = new_conv
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model



def load_effnet_state_dict(model_path, num_classes=5, device=device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict):
        for key in ("model", "state_dict", "state"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    new_state = {}
    for k, v in state.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        new_state[new_key] = v

    state = new_state

    model = build_effnet_b0(num_classes=num_classes)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)

    model = model.to(device)
    model.eval()
    print("EfficientNet-B0 model loaded successfully !!!")
    
    return model

MODEL_PATH = "efficientnet_severity_b0.pth"

model = load_effnet_state_dict(MODEL_PATH, num_classes=5, device=device)


#---------------------------------------
# 1. Load mask (grayscale)
# ---------------------------------------
def load_mask(path, size=(384,384)):
    img = Image.open(path).convert('L')
    img = img.resize(size, Image.NEAREST)
    return np.array(img).astype(np.int64)

# ---------------------------------------
# 2. Colour-map (same mapping as training)
# ---------------------------------------
def mask_to_colormap(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    color[mask == 3] = (255, 180, 180)   # medium damage
    color[mask == 4] = (255,  80,  80)   # major damage
    color[mask == 5] = (160,   0,   0)   # destruction
    color[mask == 8] = (255, 120,   0)   # blocked roads

    return color

# ---------------------------------------
# 3. Continuous damage intensity map (0..1)
# ---------------------------------------
def damage_intensity_map(mask_np):
    dmg = np.zeros_like(mask_np, dtype=np.float32)

    dmg[mask_np == 3] = 0.4   # medium damage
    dmg[mask_np == 4] = 0.7   # major damage
    dmg[mask_np == 5] = 1.0   # destruction
    dmg[mask_np == 8] = 0.6   # blocked roads

    if dmg.max() > 0:
        dmg = dmg / dmg.max()
    return dmg

# ---------------------------------------
# 4. 4-channel tensor for EfficientNet (mask + colour map)
# ---------------------------------------
def mask_and_colormap_to_tensor(mask_np):
    mask = cv2.resize(mask_np, (224,224), interpolation=cv2.INTER_NEAREST)
    cmap = mask_to_colormap(mask)

    mask_f = mask.astype(np.float32) / 255.0
    mask_f = np.expand_dims(mask_f, axis=0)           # (1,H,W)

    cmap_f = cmap.astype(np.float32) / 255.0
    cmap_f = np.transpose(cmap_f, (2,0,1))            # (3,H,W)

    x = np.concatenate([mask_f, cmap_f], axis=0)      # (4,H,W)
    x = np.expand_dims(x, axis=0)                     # (1,4,H,W)
    return torch.tensor(x, dtype=torch.float32).to(device)

# ---------------------------------------
# 5. Severity from TRAINED EfficientNet model
# ---------------------------------------
def effnet_severity(mask_np):
    model.eval()
    x = mask_and_colormap_to_tensor(mask_np)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
        sev = int(np.argmax(probs)) + 1   # 1..5
    return sev, probs

# ---------------------------------------
# 6. Damage spread from mask
# ---------------------------------------
def damage_spread(mask_np):
    total = mask_np.size
    damaged = ((mask_np==3) | (mask_np==4) | (mask_np==5) | (mask_np==8)).sum()
    ratio = damaged / total

    if   ratio < 0.05: level = 1
    elif ratio < 0.15: level = 2
    elif ratio < 0.35: level = 3
    elif ratio < 0.55: level = 4
    else:              level = 5
    return level, ratio

# ---------------------------------------
# 7. Impact matrix (severity × spread)
# ---------------------------------------
def draw_impact_matrix(sev, spread, show_colorbar=True):
    severities  = np.arange(1,6)
    spreads     = np.arange(1,6)
    impact      = np.outer(spreads, severities)  # 5x5

    im = plt.imshow(impact, cmap='RdPu', origin='lower')
    plt.title("Disaster Response Impact Matrix")
    plt.xlabel("Damage Severity Level")
    plt.ylabel("Damage Spread Level")

    for i in range(5):
        for j in range(5):
            plt.text(j, i, int(impact[i, j]),
                     ha='center', va='center', fontsize=9)

    rect = plt.Rectangle((sev-1-0.5, spread-1-0.5), 1, 1, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(rect)
    if show_colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Impact Score (Severity × Spread)")
    return impact

# ---------------------------------------
# 8. Full pipeline: visualize + export JSON
# ---------------------------------------
def analyze_and_export(mask_path, json_name=None):
    mask_np = load_mask(mask_path)
    dmg_map = damage_intensity_map(mask_np)
    severity_level, probs = effnet_severity(mask_np)
    spread_level, spread_ratio = damage_spread(mask_np)
    impact_score = int(severity_level * spread_level)

   
    impact_matrix = draw_impact_matrix(severity_level, spread_level, show_colorbar=True)


    print(f"Severity Level: {severity_level} / 5")
    print(f"Damage Spread Level: {spread_level} / 5")

    out_root= "severity_output"
    base = os.path.splitext(os.path.basename(mask_path))[0]
    if json_name is None:
        json_name = os.path.join(out_root, base)

    result = {
        "filename": os.path.basename(mask_path),
        "severity_level": int(severity_level),
        "spread_percentage": round(spread_ratio * 100, 2)
    }

    with open(json_name + ".json", "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"Saved JSON file → {json_name}.json")
   
    cmap = mask_to_colormap(mask_np)
    Image.fromarray(cmap).save(f"{json_name}_color_mask.png")
    print(f"Saved Colour Mask → {json_name}_color_mask.png")
	
    plt.figure(figsize=(6,6))
    im = plt.imshow(dmg_map, cmap='Reds', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{json_name}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Heatmap → {json_name}_heatmap.png")

    plt.figure(figsize=(6,6))
    fig = draw_impact_matrix(severity_level, spread_level, show_colorbar=True)
    plt.savefig(f"{json_name}_impact_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Impact Matrix → {json_name}_impact_matrix.png")

    return json_name, result


"""
Carbon Fibre Segmentation — HTML REPORT GENERATOR
====================================================

Updated to match test_cl.py exactly, so the Dice/IoU numbers shown in
this report are identical to what test_cl.py reports for the same
images — no more drift between the two.

Changes from the old version:
  1. encoder_name / IMG_SIZE are read from the checkpoint instead of
     being hardcoded ("efficientnet-b1" was stale — it doesn't match
     the b2 encoder the current training script saves).
  2. TTA (test-time augmentation) is applied during inference, matching
     test_cl.py and the training script's run_inference(). Without
     this, the report would show worse numbers than your real
     inference pipeline produces.
  3. Dice/IoU computed via smp.metrics (same as test_cl.py) instead of
     manual NumPy intersection/union — guarantees identical numbers
     between the two scripts for the same image.
  4. Images/report are sorted worst-to-best Dice in the HTML output,
     so the hardest cases are immediately visible at the top instead
     of buried alphabetically.
"""

import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================================
# CONFIG
# ==========================================================

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

TEST_IMG_DIR  = "test/images"
TEST_MASK_DIR = "test/masks"

OUTPUT_DIR  = "report_images"
REPORT_FILE = "report.html"

CHECKPOINT_PATH = "best_checkpoint.pth"

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

THRESHOLD = 0.5
USE_TTA   = True   # matches test_cl.py default; set False for faster single-pass

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)


# ==========================================================
# LOAD CHECKPOINT — read encoder/img_size back instead of hardcoding
# ==========================================================

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

IMG_SIZE     = checkpoint.get("img_size", 384)
ENCODER_NAME = checkpoint.get("encoder_name", "efficientnet-b2")

print(f"Using IMG_SIZE={IMG_SIZE}, encoder={ENCODER_NAME} (from checkpoint)")
print(f"Checkpoint used_ema={checkpoint.get('used_ema', 'unknown')}")


# ==========================================================
# MODEL — architecture read from checkpoint, always matches training
# ==========================================================

model = smp.UnetPlusPlus(
    encoder_name=ENCODER_NAME,
    encoder_weights=None,   # weights come from checkpoint, not ImageNet
    in_channels=3,
    classes=1,
    activation=None
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"  (checkpoint val_dice={checkpoint.get('val_dice', float('nan')):.4f}, "
      f"val_iou={checkpoint.get('val_iou', float('nan')):.4f})")


# ==========================================================
# TRANSFORM
# ==========================================================

transform = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


# ==========================================================
# TTA INFERENCE — identical logic to test_cl.py / training script
# ==========================================================

@torch.no_grad()
def predict_with_tta(image_tensor, use_tta=USE_TTA):
    """
    image_tensor: (1, 3, H, W) already on DEVICE
    Returns: (H, W) numpy probability map
    """
    def infer(x):
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            return torch.sigmoid(model(x))

    probs = [infer(image_tensor)]

    if use_tta:
        probs.append(torch.flip(infer(torch.flip(image_tensor, dims=[3])), dims=[3]))
        probs.append(torch.flip(infer(torch.flip(image_tensor, dims=[2])), dims=[2]))
        probs.append(torch.flip(infer(torch.flip(image_tensor, dims=[2, 3])), dims=[2, 3]))

    avg_prob = torch.stack(probs, dim=0).mean(dim=0)
    return avg_prob.squeeze().cpu().numpy()


# ==========================================================
# METRICS — identical to test_cl.py, guarantees matching numbers
# ==========================================================

@torch.no_grad()
def compute_metrics_from_numpy(pred_binary, mask_binary):
    """
    pred_binary, mask_binary: numpy arrays, uint8, values 0/1, same shape (H, W)
    Returns (iou, dice) via smp.metrics for exact parity with test_cl.py.
    """
    pred_t = torch.from_numpy(pred_binary).long().unsqueeze(0).unsqueeze(0)
    mask_t = torch.from_numpy(mask_binary).long().unsqueeze(0).unsqueeze(0)
    tp, fp, fn, tn = smp.metrics.get_stats(pred_t, mask_t, mode="binary")
    iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
    dice = smp.metrics.f1_score( tp, fp, fn, tn, reduction="macro").item()
    return iou, dice


# ==========================================================
# HTML HEADER
# ==========================================================

html_head = """
<html>
<head>

<title>Carbon Fibre Segmentation Report</title>

<style>

body{
    font-family:Arial;
    margin:20px;
}

table{
    border-collapse:collapse;
    margin-bottom:40px;
}

table,th,td{
    border:1px solid black;
    padding:8px;
}

.row{
    margin-bottom:60px;
}

.images{
    display:flex;
    gap:20px;
    flex-wrap:wrap;
}

.images img{
    border:1px solid black;
}

.metric{
    font-size:18px;
    margin-bottom:10px;
}

.metric.low{
    color:#b30000;
}

.metric.high{
    color:#0a7a0a;
}

</style>

</head>

<body>

<h1>Carbon Fibre Bundle Segmentation Report</h1>
<p>TTA enabled: """ + str(USE_TTA) + """</p>

"""


# ==========================================================
# PROCESS TEST IMAGES
# ==========================================================

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def find_mask_path(mask_dir, file_name):
    direct = os.path.join(mask_dir, file_name)
    if os.path.exists(direct):
        return direct
    base = os.path.splitext(file_name)[0]
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(mask_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None


files = sorted([
    f for f in os.listdir(TEST_IMG_DIR)
    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
])

results = []   # collects dicts; HTML is built after sorting by Dice

for idx, file_name in enumerate(files):

    img_path = os.path.join(TEST_IMG_DIR, file_name)

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  [skip] could not read image: {file_name}")
            continue
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_path = find_mask_path(TEST_MASK_DIR, file_name)
    if mask_path is None:
        print(f"  [skip] no matching mask for: {file_name}")
        continue

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"  [skip] could not read mask: {file_name}")
        continue

    # square resize — same as training/test_cl.py, preserves circular shape
    image   = cv2.resize(image,   (IMG_SIZE, IMG_SIZE))
    gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    gt_mask_binary = (gt_mask > 0).astype(np.uint8)

    transformed  = transform(image=image, mask=gt_mask_binary)
    image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    prob        = predict_with_tta(image_tensor, use_tta=USE_TTA)
    pred_binary = (prob > THRESHOLD).astype(np.uint8)

    iou, dice = compute_metrics_from_numpy(pred_binary, gt_mask_binary)

    pred_mask_255 = pred_binary * 255

    overlay = image.copy()
    overlay[pred_binary == 1] = [255, 0, 0]

    original_file = f"{idx}_original.png"
    gt_file       = f"{idx}_gt.png"
    pred_file     = f"{idx}_pred.png"
    overlay_file  = f"{idx}_overlay.png"

    cv2.imwrite(os.path.join(OUTPUT_DIR, original_file),
               cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, gt_file), gt_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, pred_file), pred_mask_255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, overlay_file),
               cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    results.append({
        "file_name": file_name,
        "dice": dice,
        "iou": iou,
        "original_file": original_file,
        "gt_file": gt_file,
        "pred_file": pred_file,
        "overlay_file": overlay_file,
    })

    print(f"  [{idx+1}/{len(files)}] {file_name}  Dice={dice:.4f}  IoU={iou:.4f}")


# ==========================================================
# SORT WORST -> BEST so hardest cases appear first in the report
# ==========================================================

results_sorted = sorted(results, key=lambda r: r["dice"])

html = html_head

for r in results_sorted:
    metric_class = "low" if r["dice"] < 0.6 else ("high" if r["dice"] >= 0.85 else "")
    html += f"""
    <div class='row'>

        <h2>{r['file_name']}</h2>

        <div class='metric {metric_class}'>
            Dice Score: {r['dice']:.4f}
            <br>
            IoU Score: {r['iou']:.4f}
        </div>

        <div class='images'>

            <div>
                <h3>Original</h3>
                <img src='{OUTPUT_DIR}/{r["original_file"]}' width='300'>
            </div>

            <div>
                <h3>Ground Truth</h3>
                <img src='{OUTPUT_DIR}/{r["gt_file"]}' width='300'>
            </div>

            <div>
                <h3>Prediction</h3>
                <img src='{OUTPUT_DIR}/{r["pred_file"]}' width='300'>
            </div>

            <div>
                <h3>Overlay</h3>
                <img src='{OUTPUT_DIR}/{r["overlay_file"]}' width='300'>
            </div>

        </div>

    </div>
    """


# ==========================================================
# SUMMARY
# ==========================================================

if len(results) == 0:
    print("\nNo valid image/mask pairs were found — check TEST_IMG_DIR / TEST_MASK_DIR paths.")
    mean_dice = float("nan")
    mean_iou  = float("nan")
    best      = {"file_name": "N/A", "dice": float("nan")}
    worst     = {"file_name": "N/A", "dice": float("nan")}
else:
    all_dice  = [r["dice"] for r in results]
    all_iou   = [r["iou"]  for r in results]
    mean_dice = np.mean(all_dice)
    mean_iou  = np.mean(all_iou)
    best      = max(results, key=lambda r: r["dice"])
    worst     = min(results, key=lambda r: r["dice"])

summary = f"""
<h2>Summary</h2>

<table>

<tr>
<th>Total Test Images</th>
<td>{len(results)}</td>
</tr>

<tr>
<th>TTA Enabled</th>
<td>{USE_TTA}</td>
</tr>

<tr>
<th>Mean Dice</th>
<td>{mean_dice:.4f}</td>
</tr>

<tr>
<th>Mean IoU</th>
<td>{mean_iou:.4f}</td>
</tr>

<tr>
<th>Best Dice Image</th>
<td>{best['file_name']} ({best['dice']:.4f})</td>
</tr>

<tr>
<th>Worst Dice Image</th>
<td>{worst['file_name']} ({worst['dice']:.4f})</td>
</tr>

</table>
"""

html = html.replace("<h1>Carbon Fibre Bundle Segmentation Report</h1>",
                    "<h1>Carbon Fibre Bundle Segmentation Report</h1>" + summary)

html += "</body></html>"

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print()
print("====================================")
print("Report Generated Successfully")
print("====================================")
print(f"Mean Dice : {mean_dice:.4f}")
print(f"Mean IoU  : {mean_iou:.4f}")
print(f"Saved HTML: {REPORT_FILE}")
print(f"Images    : {OUTPUT_DIR}/")
"""
Carbon Fibre Segmentation — HTML REPORT GENERATOR (K-FOLD ENSEMBLE)
========================================================================

Companion report for the k-fold training script (UnetPlusPlusDeepSup +
EMA/SWA + 5-fold CV + boundary loss + deep supervision). This mirrors
that script's actual test-time inference path exactly, so the Dice/IoU
numbers shown here are identical to what evaluate_ensemble_on_test_set()
in the training script reports for the same images — no drift between
"the report" and "the real pipeline".

What this matches from the training script, and why each one matters:

  1. MODEL ARCHITECTURE: UnetPlusPlusDeepSup (encoder + decoder + seg
     head + 3 lazily-built aux 1x1 conv heads on the deepest encoder
     features). NOT plain smp.UnetPlusPlus — that has no aux_heads.*
     keys, so it would fail to load these checkpoints' state_dict (or
     silently load a different model if you set strict=False).

  2. ENSEMBLE, not a single best_checkpoint.pth: every fold*_best.pth
     in checkpoints_kfold/ is auto-discovered and averaged in
     probability space, same as ensemble_predict().

  3. MULTI-SCALE + FLIP TTA: 0.9x / 1.0x / 1.1x scales x 4 flip
     variants per fold, identical to _predict_prob_map(). The old
     single-scale h/v/180-flip-only TTA would systematically
     under-predict on bundles that are bigger/smaller than IMG_SIZE
     after the dataset's fixed resize.

  4. Per-checkpoint model load/predict/delete loop: each fold's model
     is built fresh, used, then discarded before moving to the next
     fold and the next image — same memory footprint as the training
     script's 4GB-VRAM-budgeted ensemble_predict(). Slower than
     preloading all 5 folds, but matches the script's stated GPU
     constraint exactly rather than silently using more VRAM.

  5. CIRCULARITY POST-FILTER (enforce_circular_shape), applied AFTER
     averaging across folds+TTA, same default area/circularity
     thresholds as the training script.

  6. METRICS: plain intersection-over-union / Dice via NumPy, same
     formula as evaluate_ensemble_on_test_set() (NOT smp.metrics —
     the k-fold script's own test-set evaluator uses manual NumPy, so
     that's the ground truth this report has to match).

  7. encoder_name / img_size are read PER FOLD CHECKPOINT, never
     hardcoded — folds trained with different encoders/img sizes
     (if you ever change the config between runs) still load correctly.

  8. Images/report sorted worst-to-best Dice, hardest cases first.
"""

import os
import glob
import cv2
import torch
import torch.nn as nn
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

CHECKPOINT_DIR     = "checkpoints_kfold"
CHECKPOINT_PATTERN = "fold*_best.pth"

OUTPUT_DIR  = "report_images"
REPORT_FILE = "report.html"

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

THRESHOLD                = 0.5
TTA_SCALES                = (0.9, 1.0, 1.1)
USE_FLIPS                 = True
APPLY_CIRCULARITY_FILTER  = True   # matches evaluate_ensemble_on_test_set() default

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)


# ==========================================================
# DISCOVER FOLD CHECKPOINTS
# ==========================================================

CHECKPOINT_PATHS = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, CHECKPOINT_PATTERN)))
if len(CHECKPOINT_PATHS) == 0:
    raise FileNotFoundError(
        f"No checkpoints matching '{CHECKPOINT_PATTERN}' found in "
        f"'{CHECKPOINT_DIR}/'. Run the k-fold training script first."
    )

print(f"Found {len(CHECKPOINT_PATHS)} fold checkpoint(s):")
FOLD_META = []
for p in CHECKPOINT_PATHS:
    meta = torch.load(p, map_location="cpu")
    FOLD_META.append({
        "path"        : p,
        "fold"        : meta.get("fold"),
        "epoch"       : meta.get("epoch"),
        "val_dice"    : meta.get("val_dice"),
        "variant"     : meta.get("used_variant"),
        "encoder_name": meta.get("encoder_name", "efficientnet-b2"),
        "img_size"    : meta.get("img_size", 512),
    })
    del meta
    print(f"  fold={FOLD_META[-1]['fold']}  epoch={FOLD_META[-1]['epoch']}  "
          f"val_dice={FOLD_META[-1]['val_dice']:.4f}  "
          f"variant={FOLD_META[-1]['variant']}")


# ==========================================================
# MODEL — identical to the training script's UnetPlusPlusDeepSup so
# fold checkpoints (including the aux_heads.* keys) load cleanly.
# ==========================================================

class UnetPlusPlusDeepSup(nn.Module):
    def __init__(self, encoder_name="efficientnet-b2", encoder_weights=None):
        super().__init__()
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.encoder  = self.base.encoder
        self.decoder  = self.base.decoder
        self.seg_head = self.base.segmentation_head

        self.aux_heads = nn.ModuleList()
        self._aux_built = False

    def _build_aux_heads(self, sample_feats):
        channels = [f.shape[1] for f in sample_feats]
        for c in channels:
            self.aux_heads.append(nn.Conv2d(c, 1, kernel_size=1))
        self.aux_heads = self.aux_heads.to(sample_feats[0].device)
        self._aux_built = True

    def forward(self, x):
        target_size = x.shape[-2:]
        feats = self.encoder(x)

        decoder_output = self.decoder(feats)
        final_mask = self.seg_head(decoder_output)

        tapped = feats[-3:] if len(feats) >= 3 else feats
        if not self._aux_built:
            self._build_aux_heads(tapped)

        aux_outputs = []
        for feat, head in zip(tapped, self.aux_heads):
            aux = head(feat)
            aux = nn.functional.interpolate(
                aux, size=target_size, mode="bilinear", align_corners=False
            )
            aux_outputs.append(aux)

        return final_mask, aux_outputs


def build_model_from_checkpoint(checkpoint_path):
    """
    Builds a fresh model, triggers the lazy aux-head build with a dummy
    forward pass (so the state_dict's aux_heads.* keys have somewhere
    to land), then loads weights. Mirrors
    training_script._build_model_from_checkpoint() exactly.
    """
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    encoder_name = ckpt.get("encoder_name", "efficientnet-b2")
    img_size     = ckpt.get("img_size", 512)

    model = UnetPlusPlusDeepSup(encoder_name=encoder_name, encoder_weights=None)
    model = model.to(DEVICE)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size).to(DEVICE)
        model(dummy)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, img_size


# ==========================================================
# MULTI-SCALE + FLIP TTA — identical to _predict_prob_map() in the
# training script.
# ==========================================================

_tta_transform = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


@torch.no_grad()
def predict_prob_map(model, image_rgb, img_size, scales=TTA_SCALES, use_flips=USE_FLIPS):
    all_probs = []

    for scale in scales:
        size = max(64, int(img_size * scale))
        resized = cv2.resize(image_rgb, (size, size))
        resized = cv2.resize(resized, (img_size, img_size))
        tensor  = _tta_transform(image=resized)["image"].unsqueeze(0).to(DEVICE)

        variants = [tensor]
        if use_flips:
            variants.append(torch.flip(tensor, dims=[3]))            # h-flip
            variants.append(torch.flip(tensor, dims=[2]))            # v-flip
            variants.append(torch.flip(tensor, dims=[2, 3]))         # 180 rot

        for i, v in enumerate(variants):
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                final_logits, _ = model(v)
                prob = torch.sigmoid(final_logits)
            if i == 1:
                prob = torch.flip(prob, dims=[3])
            elif i == 2:
                prob = torch.flip(prob, dims=[2])
            elif i == 3:
                prob = torch.flip(prob, dims=[2, 3])
            all_probs.append(prob.squeeze().cpu().numpy())

    return np.mean(all_probs, axis=0)


def enforce_circular_shape(pred_mask_np, min_area=50, circularity_thresh=0.55):
    """Identical to the training script's post-filter."""
    contours, _ = cv2.findContours(
        pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    clean_mask = np.zeros_like(pred_mask_np)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or area < min_area:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= circularity_thresh:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
    return clean_mask


def ensemble_predict(image_rgb, threshold=THRESHOLD,
                      apply_circularity_filter=APPLY_CIRCULARITY_FILTER):
    """
    Loads each fold checkpoint fresh, runs multi-scale+flip TTA, deletes
    the model, moves to the next fold — same load/predict/delete loop
    (and same 4GB-VRAM footprint) as ensemble_predict() in the training
    script. Averages probabilities across all folds, thresholds, resizes
    back to the original image size, then optionally applies the
    circularity filter — same order of operations.
    """
    orig_h, orig_w = image_rgb.shape[:2]

    all_fold_probs = []
    for ckpt_path in CHECKPOINT_PATHS:
        model, img_size = build_model_from_checkpoint(ckpt_path)
        prob = predict_prob_map(model, image_rgb, img_size)
        all_fold_probs.append(prob)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    avg_prob = np.mean(all_fold_probs, axis=0)
    binary   = (avg_prob > threshold).astype(np.uint8) * 255
    binary   = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if apply_circularity_filter:
        binary = enforce_circular_shape(binary)

    return binary, avg_prob


# ==========================================================
# METRICS — identical formula to evaluate_ensemble_on_test_set() in
# the training script (manual NumPy intersection/union, NOT
# smp.metrics), so these numbers can never drift from that function.
# ==========================================================

def compute_metrics(pred_binary_01, gt_binary_01):
    intersection = np.logical_and(gt_binary_01, pred_binary_01).sum()
    union        = np.logical_or(gt_binary_01, pred_binary_01).sum()
    iou  = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (gt_binary_01.sum() + pred_binary_01.sum()) \
           if (gt_binary_01.sum() + pred_binary_01.sum()) > 0 else 1.0
    return iou, dice


# ==========================================================
# HTML HEADER
# ==========================================================

fold_rows = "".join(
    f"<tr><td>{m['fold']}</td><td>{m['epoch']}</td>"
    f"<td>{m['val_dice']:.4f}</td><td>{m['variant']}</td>"
    f"<td>{m['encoder_name']}</td><td>{m['img_size']}</td></tr>"
    for m in FOLD_META
)

html_head = """
<html>
<head>

<title>Carbon Fibre Segmentation Report — K-Fold Ensemble</title>

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

<h1>Carbon Fibre Bundle Segmentation Report — K-Fold Ensemble</h1>

<p>Folds ensembled: """ + str(len(CHECKPOINT_PATHS)) + """</p>
<p>TTA: multi-scale """ + str(TTA_SCALES) + """ &times; flips=""" + str(USE_FLIPS) + """</p>
<p>Circularity post-filter: """ + str(APPLY_CIRCULARITY_FILTER) + """</p>

<h2>Fold checkpoints used</h2>
<table>
<tr><th>Fold</th><th>Epoch</th><th>Val Dice (saved)</th>
<th>Variant</th><th>Encoder</th><th>Img Size</th></tr>
""" + fold_rows + """
</table>

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


def load_image_raw(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


files = sorted([
    f for f in os.listdir(TEST_IMG_DIR)
    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
])

results = []   # collects dicts; HTML is built after sorting by Dice

for idx, file_name in enumerate(files):

    img_path = os.path.join(TEST_IMG_DIR, file_name)

    image_rgb = load_image_raw(img_path)
    if image_rgb is None:
        print(f"  [skip] could not read image: {file_name}")
        continue

    mask_path = find_mask_path(TEST_MASK_DIR, file_name)
    if mask_path is None:
        print(f"  [skip] no matching mask for: {file_name}")
        continue

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"  [skip] could not read mask: {file_name}")
        continue

    orig_h, orig_w = image_rgb.shape[:2]
    gt_mask   = cv2.resize(gt_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    pred_mask_255, _ = ensemble_predict(image_rgb)
    pred_binary = (pred_mask_255 > 0).astype(np.uint8)

    iou, dice = compute_metrics(pred_binary, gt_binary)

    overlay = image_rgb.copy()
    overlay[pred_binary == 1] = [255, 0, 0]

    original_file = f"{idx}_original.png"
    gt_file       = f"{idx}_gt.png"
    pred_file     = f"{idx}_pred.png"
    overlay_file  = f"{idx}_overlay.png"

    cv2.imwrite(os.path.join(OUTPUT_DIR, original_file),
               cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
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

    if DEVICE == "cuda":
        torch.cuda.empty_cache()


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
    std_dice  = float("nan")
    std_iou   = float("nan")
    best      = {"file_name": "N/A", "dice": float("nan")}
    worst     = {"file_name": "N/A", "dice": float("nan")}
else:
    all_dice  = [r["dice"] for r in results]
    all_iou   = [r["iou"]  for r in results]
    mean_dice = np.mean(all_dice)
    mean_iou  = np.mean(all_iou)
    std_dice  = np.std(all_dice)
    std_iou   = np.std(all_iou)
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
<th>Folds Ensembled</th>
<td>{len(CHECKPOINT_PATHS)}</td>
</tr>

<tr>
<th>TTA</th>
<td>scales={TTA_SCALES}, flips={USE_FLIPS}</td>
</tr>

<tr>
<th>Circularity Filter</th>
<td>{APPLY_CIRCULARITY_FILTER}</td>
</tr>

<tr>
<th>Mean Dice</th>
<td>{mean_dice:.4f} &plusmn; {std_dice:.4f}</td>
</tr>

<tr>
<th>Mean IoU</th>
<td>{mean_iou:.4f} &plusmn; {std_iou:.4f}</td>
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

html = html.replace(
    "<h1>Carbon Fibre Bundle Segmentation Report — K-Fold Ensemble</h1>",
    "<h1>Carbon Fibre Bundle Segmentation Report — K-Fold Ensemble</h1>" + summary
)

html += "</body></html>"

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print()
print("====================================")
print("Report Generated Successfully")
print("====================================")
print(f"Mean Dice : {mean_dice:.4f} +/- {std_dice:.4f}")
print(f"Mean IoU  : {mean_iou:.4f} +/- {std_iou:.4f}")
print(f"Saved HTML: {REPORT_FILE}")
print(f"Images    : {OUTPUT_DIR}/")
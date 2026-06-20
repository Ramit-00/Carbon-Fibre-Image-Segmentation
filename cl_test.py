"""
Carbon Fibre Bundle Segmentation — TEST SCRIPT
================================================

Matches the high-capacity training script exactly. Key change from
before: encoder_name and IMG_SIZE are no longer hardcoded here — they
are read directly from the checkpoint (the training script now saves
both), so this file can never silently drift out of sync if you change
the encoder or image size in training later.

Also supports TTA (test-time augmentation) so the reported Dice/IoU
match what run_inference()/run_inference_folder() will actually produce
in the training script, not a lower TTA-off number.
"""

import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================================
# CONFIG
# =====================================

TEST_IMG_DIR  = "test/images"
TEST_MASK_DIR = "test/masks"

CHECKPOINT_PATH = "best_checkpoint.pth"

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

THRESHOLD = 0.5
USE_TTA   = True     # set False for faster single-pass testing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")
print("Device:", DEVICE)


# =====================================
# LOAD CHECKPOINT — read encoder/img_size back instead of hardcoding
# =====================================

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

IMG_SIZE     = checkpoint.get("img_size", 384)
ENCODER_NAME = checkpoint.get("encoder_name", "efficientnet-b2")

print(f"Using IMG_SIZE={IMG_SIZE}, encoder={ENCODER_NAME} (from checkpoint)")
print(f"Checkpoint used_ema={checkpoint.get('used_ema', 'unknown')}")


# =====================================
# MODEL — architecture is read from the checkpoint, always matches training
# =====================================

model = smp.UnetPlusPlus(
    encoder_name=ENCODER_NAME,
    encoder_weights=None,     # weights come from checkpoint, not ImageNet
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


# =====================================
# TRANSFORM
# =====================================

transform = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


# =====================================
# TTA INFERENCE — must match training script's _predict_with_tta exactly
# =====================================

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


# =====================================
# METRICS — identical to training script's batch_metrics()
# =====================================

@torch.no_grad()
def compute_metrics_from_numpy(pred_binary, mask_binary):
    """
    pred_binary, mask_binary: numpy arrays, uint8, values 0/1, same shape (H, W)
    Returns (iou, dice) via smp.metrics for exact parity with train/val numbers.
    """
    pred_t = torch.from_numpy(pred_binary).long().unsqueeze(0).unsqueeze(0)
    mask_t = torch.from_numpy(mask_binary).long().unsqueeze(0).unsqueeze(0)
    tp, fp, fn, tn = smp.metrics.get_stats(pred_t, mask_t, mode="binary")
    iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
    dice = smp.metrics.f1_score( tp, fp, fn, tn, reduction="macro").item()
    return iou, dice


# =====================================
# TEST LOOP
# =====================================

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


dice_scores = []
iou_scores  = []
per_image_results = []   # (filename, dice, iou)

files = sorted([
    f for f in os.listdir(TEST_IMG_DIR)
    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
])

with torch.no_grad():

    for file_name in files:

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

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [skip] could not read mask: {file_name}")
            continue

        # square resize — same as training, preserves circular shape
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask  = cv2.resize(mask,  (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.uint8)

        transformed = transform(image=image, mask=mask)
        image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

        prob = predict_with_tta(image_tensor, use_tta=USE_TTA)
        pred_binary = (prob > THRESHOLD).astype(np.uint8)

        iou, dice = compute_metrics_from_numpy(pred_binary, mask)

        dice_scores.append(dice)
        iou_scores.append(iou)
        per_image_results.append((file_name, dice, iou))


# =====================================
# RESULTS
# =====================================

print("\n========== TEST RESULTS ==========")
print(f"TTA enabled     : {USE_TTA}")

if len(dice_scores) == 0:
    print("No valid image/mask pairs were found — check your TEST_IMG_DIR / TEST_MASK_DIR paths.")
else:
    print(f"Mean Dice Score : {np.mean(dice_scores):.4f}")
    print(f"Mean IoU Score  : {np.mean(iou_scores):.4f}")
    print(f"Images Tested   : {len(dice_scores)}")

    worst = sorted(per_image_results, key=lambda x: x[1])[:5]
    print("\nWorst 5 by Dice score:")
    for fname, dice, iou in worst:
        print(f"  {fname:30s}  Dice={dice:.4f}  IoU={iou:.4f}")

    best = sorted(per_image_results, key=lambda x: x[1], reverse=True)[:5]
    print("\nBest 5 by Dice score:")
    for fname, dice, iou in best:
        print(f"  {fname:30s}  Dice={dice:.4f}  IoU={iou:.4f}")
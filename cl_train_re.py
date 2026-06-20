"""
Carbon Fibre Bundle Segmentation — CONTINUATION / FINE-TUNE SCRIPT
==================================================================

PURPOSE
-------
You already have a checkpoint ("best_checkpoint.pth") trained on ~79
original images, scoring:
    mean Dice = 0.9109
    mean IoU  = 0.8380

You now have 5 additional images (+ masks). This script:

  1. Loads the EXISTING model weights from best_checkpoint.pth
     (architecture must match: UnetPlusPlus / efficientnet-b2).
  2. Continues training on the COMBINED dataset (79 original + 5 new
     images, all in the same train_original/images + masks folders —
     just drop the new files in there before running this script).
  3. Saves ALL new checkpoints to a SEPARATE file: best_checkpoint_re.pth
     -> best_checkpoint.pth is NEVER touched, NEVER overwritten.
  4. Before training starts, evaluates the ORIGINAL checkpoint on your
     current validation set so you have an honest baseline to compare
     the fine-tuned result against (your 0.9109/0.8380 numbers may have
     come from a different val split — this gives you a same-split
     baseline).

WHY THESE SETTINGS (vs. the original from-scratch script)
-----------------------------------------------------------
  - Fresh optimizer/EMA/scheduler (NOT resumed from the old run).
    best_checkpoint.pth only stores model + optimizer state, not
    scheduler/EMA state — and resuming Adam momentum tuned for a
    79-image regime onto a 84-image regime at the OLD learning rate
    risks a destabilizing jump. A fresh, low-LR optimizer is safer.
  - LR is 10x lower than the original run (1e-4 vs 1e-3), with a
    shorter warm-restart cycle (T0=5 vs 10). This is fine-tuning: you
    want to nudge the model toward the new images, not re-explore the
    loss landscape aggressively the way a from-scratch run should.
  - Fewer max epochs (60) + tighter patience (15). 5 new images need
    far less training to be absorbed than a full run from scratch.
  - The script keeps EMA + warm restarts + the same loss / augment
    pipeline as your original successful run, just turned down in
    intensity, so the only real change is the new data + gentler LR.

WORKFLOW
--------
  1. Add your 5 new images to:      train_original/images/
     Add their masks to:            train_original/masks/
     (same naming convention as before — image name maps to mask name)
  2. Make sure best_checkpoint.pth (the original) is in this directory.
  3. Run this script:  python continue_training.py
  4. Compare best_checkpoint_re.pth's val Dice/IoU (printed at the end)
     against the ORIGINAL baseline (also printed, evaluated on the
     same val set right before training starts).
  5. Only switch to best_checkpoint_re.pth if it's actually better —
     this script will never overwrite best_checkpoint.pth, so you can
     always fall back.
"""

import os
import random
import copy
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ══════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

TRAIN_IMG_DIR  = "train_original/images"   # 79 original + 5 new = 84 images
TRAIN_MASK_DIR = "train_original/masks"
VAL_IMG_DIR    = "validation/images"
VAL_MASK_DIR   = "validation/masks"

# ── checkpoint paths — kept strictly separate ───────────────────────
SOURCE_CHECKPOINT = "best_checkpoint.pth"      # READ ONLY. Never written to.
OUTPUT_CHECKPOINT = "best_checkpoint_re.pth"   # All new saves go here.

IMG_SIZE   = 512
BATCH_SIZE = 8
GRAD_ACCUM = 8              # effective batch = 8 x 8 = 64

# ── fine-tune intensity: turned DOWN relative to the original
#    from-scratch run, since we're nudging an already-good model ──
EPOCHS     = 100              # was 150 — 5 new images don't need that long
LR         = 1e-4            # was 1e-3 — 10x lower, safer for fine-tuning
MIN_LR     = 1e-6
PATIENCE   = 20              # was 30 — tighter, since less should be needed

RESTART_T0     = 5           # was 10 — shorter, gentler exploration bursts
RESTART_T_MULT = 2

POS_WEIGHT = 4.0
EMA_DECAY  = 0.999

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

NUM_WORKERS = 4

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════════
# AUGMENTATIONS  (unchanged from the original successful run)
# ══════════════════════════════════════════════════════════════════

train_transform = A.Compose([
    A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.3),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(0.0, 0.05),
        rotate=0,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.4
    ),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(noise_scale_factor=0.08, p=0.3),
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(8, 28),
        hole_width_range=(8, 28),
        fill_value=0,
        p=0.25
    ),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


# ══════════════════════════════════════════════════════════════════
# DATASET  — with RAM cache
# ══════════════════════════════════════════════════════════════════

class CarbonFibreDataset(Dataset):
    """
    Loads + resizes all images/masks into RAM at __init__ time, then
    applies augmentation live from RAM every __getitem__ call.
    Works the same whether the folder has 79 or 84 images — just
    drop your 5 new image+mask pairs into the existing folders.
    """

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, img_dir, mask_dir, transform=None, img_size=IMG_SIZE):
        self.transform = transform
        self.img_size  = img_size

        all_files = sorted(os.listdir(img_dir))
        self.files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTS
        ]
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        print(f"  Caching {len(self.files)} images from {img_dir} ...")
        self.cache_images = []
        self.cache_masks  = []

        for name in self.files:
            img_path = os.path.join(img_dir, name)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise FileNotFoundError(f"Cannot read image: {img_path}")
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_size, img_size))

            mask_path = self._find_mask_path(mask_dir, name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot read mask: {mask_path}")
            mask = cv2.resize(mask, (img_size, img_size),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)

            self.cache_images.append(image)
            self.cache_masks.append(mask)

        print(f"  Cached {len(self.cache_images)} image/mask pairs in RAM.")

    def _find_mask_path(self, mask_dir, name):
        direct = os.path.join(mask_dir, name)
        if os.path.exists(direct):
            return direct
        base = os.path.splitext(name)[0]
        for ext in self.SUPPORTED_EXTS:
            candidate = os.path.join(mask_dir, base + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No matching mask for {name} in {mask_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.cache_images[idx].copy()
        mask  = self.cache_masks[idx].copy()

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"]
        else:
            image = torch.from_numpy(
                (image.astype(np.float32) / 255.0).transpose(2, 0, 1)
            )
            mask = torch.from_numpy(mask.astype(np.float32))

        mask = mask.float().unsqueeze(0)
        return image, mask


def create_loaders():
    train_ds = CarbonFibreDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                  transform=train_transform, img_size=IMG_SIZE)
    val_ds   = CarbonFibreDataset(VAL_IMG_DIR,   VAL_MASK_DIR,
                                  transform=val_transform,   img_size=IMG_SIZE)

    print(f"Train images (old + new) : {len(train_ds)}")
    print(f"Validation images        : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(NUM_WORKERS, 2),
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

def build_model(encoder_name="efficientnet-b2", encoder_weights="imagenet"):
    model = smp.UnetPlusPlus(
        encoder_name    = encoder_name,
        encoder_weights = encoder_weights,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {n_params:,}")
    return model


def load_source_checkpoint(checkpoint_path):
    """
    Loads ONLY the model weights from the existing checkpoint.
    Architecture is rebuilt fresh (encoder_weights=None, since we're
    about to overwrite all weights anyway) then the state dict from
    your trained checkpoint is loaded on top.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find source checkpoint: {checkpoint_path}\n"
            f"Make sure best_checkpoint.pth is in the working directory."
        )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    encoder_name = ckpt.get("encoder_name", "efficientnet-b2")
    img_size     = ckpt.get("img_size", IMG_SIZE)

    model = build_model(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\nLoaded source checkpoint: {checkpoint_path}")
    print(f"  Original epoch    : {ckpt.get('epoch', 'unknown')}")
    print(f"  Original val Dice : {ckpt.get('val_dice', 'unknown')}")
    print(f"  Original val IoU  : {ckpt.get('val_iou', 'unknown')}")
    print(f"  Encoder            : {encoder_name}")
    print(f"  Used EMA weights   : {ckpt.get('used_ema', 'unknown')}\n")

    if img_size != IMG_SIZE:
        print(f"  WARNING: checkpoint was trained at img_size={img_size}, "
              f"but this script is set to IMG_SIZE={IMG_SIZE}. "
              f"Consider matching them.")

    return model, encoder_name


# ══════════════════════════════════════════════════════════════════
# EMA
# ══════════════════════════════════════════════════════════════════

class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1 - self.decay)
            else:
                ema_v.copy_(model_v)

    def state_dict(self):
        return self.ema.state_dict()


# ══════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════

_dice    = smp.losses.DiceLoss(mode="binary", smooth=1.0)
_tversky = smp.losses.TverskyLoss(mode="binary", alpha=0.3, beta=0.7, smooth=1.0)
_bce     = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
)

def combined_loss(logits, masks):
    return _dice(logits, masks) + _tversky(logits, masks) + 0.5 * _bce(logits, masks)


# ══════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def batch_metrics(logits, masks, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).long()
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode="binary")
    iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
    dice = smp.metrics.f1_score( tp, fp, fn, tn, reduction="macro").item()
    return iou, dice


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum = iou_sum = dice_sum = 0.0
    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE,  non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(images)
            loss   = combined_loss(logits, masks)
        loss_sum += loss.item()
        iou, dice = batch_metrics(logits, masks)
        iou_sum  += iou
        dice_sum += dice
    n = len(loader)
    return loss_sum / n, iou_sum / n, dice_sum / n


# ══════════════════════════════════════════════════════════════════
# CONTINUATION TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def continue_train():
    # ── load existing model weights (read-only source) ─────────
    model, encoder_name = load_source_checkpoint(SOURCE_CHECKPOINT)

    # ── data: same folders, just with your 5 new images added ──
    train_loader, val_loader = create_loaders()

    # ── BASELINE: evaluate the loaded (original) weights on the
    #    current val set BEFORE any fine-tuning, so you have an
    #    honest, same-split number to compare the result against ──
    print("Evaluating ORIGINAL checkpoint on current validation set "
          "(baseline, before fine-tuning)...")
    base_loss, base_iou, base_dice = evaluate(model, val_loader)
    print(f"  Baseline (original weights) -> "
          f"Val loss={base_loss:.4f}  IoU={base_iou:.4f}  Dice={base_dice:.4f}\n")

    # ── fresh optimizer / scheduler / EMA — intentionally NOT
    #    resumed from the old run (see module docstring for why) ──
    ema = ModelEMA(model, decay=EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=RESTART_T0, T_mult=RESTART_T_MULT, eta_min=MIN_LR
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # best so far starts at the baseline — we only checkpoint if we
    # actually beat what the original model could already do
    best_val_dice     = base_dice
    best_val_loss     = base_loss
    epochs_no_improve = 0
    saved_any_checkpoint = False

    for epoch in range(EPOCHS):

        # ─── train ───────────────────────────────────────────────
        model.train()
        t_loss = t_iou = t_dice = 0.0
        optimizer.zero_grad()

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(images)
                loss   = combined_loss(logits, masks) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            scheduler.step(epoch + step / len(train_loader))

            t_loss += loss.item() * GRAD_ACCUM
            iou, dice = batch_metrics(logits, masks)
            t_iou  += iou
            t_dice += dice

        n = len(train_loader)
        t_loss /= n; t_iou /= n; t_dice /= n

        # ─── validate: raw model AND EMA model ──────────────────
        v_loss, v_iou, v_dice       = evaluate(model,   val_loader)
        ema_loss, ema_iou, ema_dice = evaluate(ema.ema, val_loader)

        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Ep {epoch+1:03d}/{EPOCHS} | "
            f"Train loss={t_loss:.4f} IoU={t_iou:.4f} Dice={t_dice:.4f} | "
            f"Val loss={v_loss:.4f} IoU={v_iou:.4f} Dice={v_dice:.4f} | "
            f"EMA Dice={ema_dice:.4f} | LR={cur_lr:.2e}"
        )

        candidate_dice = max(v_dice, ema_dice)
        use_ema        = ema_dice >= v_dice

        if candidate_dice > best_val_dice:
            best_val_dice     = candidate_dice
            best_val_loss     = ema_loss if use_ema else v_loss
            epochs_no_improve = 0
            saved_any_checkpoint = True

            save_state = ema.state_dict() if use_ema else model.state_dict()
            torch.save({
                "epoch"                : epoch + 1,
                "model_state_dict"     : save_state,
                "optimizer_state_dict" : optimizer.state_dict(),
                "val_loss"             : best_val_loss,
                "val_iou"              : ema_iou if use_ema else v_iou,
                "val_dice"             : best_val_dice,
                "img_size"             : IMG_SIZE,
                "encoder_name"         : encoder_name,
                "used_ema"             : use_ema,
                "source_checkpoint"    : SOURCE_CHECKPOINT,
                "baseline_val_dice"    : base_dice,
                "baseline_val_iou"     : base_iou,
            }, OUTPUT_CHECKPOINT)
            tag = "EMA" if use_ema else "raw"
            print(f"  Checkpoint saved [{tag}] -> {OUTPUT_CHECKPOINT} "
                  f"(Val Dice={best_val_dice:.4f}, beats baseline {base_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.")
                break

    print(f"\n{'='*60}")
    print("DONE.")
    print(f"  Baseline (original best_checkpoint.pth) : "
          f"Dice={base_dice:.4f}  IoU={base_iou:.4f}")
    if saved_any_checkpoint:
        print(f"  Fine-tuned ({OUTPUT_CHECKPOINT})          : "
              f"Dice={best_val_dice:.4f}  Loss={best_val_loss:.4f}")
        print(f"  -> {OUTPUT_CHECKPOINT} IMPROVED over the original. "
              f"best_checkpoint.pth was left unchanged.")
    else:
        print(f"  No epoch beat the original baseline, so "
              f"{OUTPUT_CHECKPOINT} was NEVER WRITTEN.")
        print(f"  Your original best_checkpoint.pth remains the best model "
              f"you have — nothing to switch to.")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
# INFERENCE  (loads from best_checkpoint_re.pth by default;
#             change checkpoint_path to compare against the original)
# ══════════════════════════════════════════════════════════════════

def _load_and_preprocess(image_path, img_size):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (img_size, img_size))
    tfm     = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    tensor  = tfm(image=resized)["image"]
    return image, orig_h, orig_w, tensor


def _build_model_from_checkpoint(checkpoint_path):
    ckpt         = torch.load(checkpoint_path, map_location=DEVICE,
                              weights_only=False)
    encoder_name = ckpt.get("encoder_name", "efficientnet-b2")
    img_size     = ckpt.get("img_size", IMG_SIZE)

    model = smp.UnetPlusPlus(
        encoder_name    = encoder_name,
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}  Val Dice={ckpt['val_dice']:.4f}  "
          f"(encoder={encoder_name}, used_ema={ckpt.get('used_ema','unknown')})")
    return model, img_size


@torch.no_grad()
def _predict_with_tta(model, tensor, use_tta=True):
    tensor = tensor.unsqueeze(0).to(DEVICE)

    def infer(x):
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            return torch.sigmoid(model(x))

    probs = [infer(tensor)]
    if use_tta:
        probs.append(torch.flip(infer(torch.flip(tensor, dims=[3])), dims=[3]))
        probs.append(torch.flip(infer(torch.flip(tensor, dims=[2])), dims=[2]))
        probs.append(torch.flip(infer(torch.flip(tensor, dims=[2, 3])), dims=[2, 3]))

    return torch.stack(probs).mean(0).squeeze().cpu().numpy()


def run_inference(image_path, checkpoint_path=OUTPUT_CHECKPOINT,
                  threshold=0.5, out_dir=".", use_tta=True):
    model, img_size = _build_model_from_checkpoint(checkpoint_path)
    image, orig_h, orig_w, tensor = _load_and_preprocess(image_path, img_size)

    prob   = _predict_with_tta(model, tensor, use_tta=use_tta)
    binary = (prob > threshold).astype(np.uint8) * 255
    binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    bgr         = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    colour_mask = np.zeros_like(bgr)
    colour_mask[..., 2] = binary
    blended     = cv2.addWeighted(bgr, 0.7, colour_mask, 0.3, 0)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"),    binary)
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), blended)
    print(f"Saved: {base}_mask.png / {base}_overlay.png -> {out_dir}/")
    return binary


def run_inference_folder(folder_path, checkpoint_path=OUTPUT_CHECKPOINT,
                         threshold=0.5, out_dir="inference_results", use_tta=True):
    model, img_size = _build_model_from_checkpoint(checkpoint_path)
    os.makedirs(out_dir, exist_ok=True)

    exts  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in exts
    ])

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        image, orig_h, orig_w, tensor = _load_and_preprocess(img_path, img_size)

        prob   = _predict_with_tta(model, tensor, use_tta=use_tta)
        binary = (prob > threshold).astype(np.uint8) * 255
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        bgr         = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        colour_mask = np.zeros_like(bgr)
        colour_mask[..., 2] = binary
        blended     = cv2.addWeighted(bgr, 0.7, colour_mask, 0.3, 0)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"),    binary)
        cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), blended)
        print(f"  {fname} -> done")

    print(f"\nAll results saved to: {out_dir}/")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    continue_train()

    # run_inference("test/images/sample.png", checkpoint_path="best_checkpoint_re.pth",
    #               threshold=0.5, out_dir="results", use_tta=True)
    # run_inference_folder("test/images", checkpoint_path="best_checkpoint_re.pth",
    #                       threshold=0.5, out_dir="results", use_tta=True)
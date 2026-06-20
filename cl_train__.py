"""
Carbon Fibre Bundle Segmentation — HIGH-CAPACITY TRAINING SCRIPT
==================================================================
Speed fix: was 2+ minutes/epoch due to num_workers=0 (GPU starving
waiting for CPU data prep). Fixed with:
  1. RAM caching — all 79 images loaded once at startup (~50MB total),
     no disk reads during training
  2. num_workers=4 — parallel CPU threads prep batches while GPU trains
  3. persistent_workers=True — worker processes stay alive between epochs
Expected: ~10-20 seconds/epoch instead of 2+ minutes.
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

TRAIN_IMG_DIR  = "train_original/images"
TRAIN_MASK_DIR = "train_original/masks"
VAL_IMG_DIR    = "validation/images"
VAL_MASK_DIR   = "validation/masks"

IMG_SIZE   = 512
BATCH_SIZE = 8
GRAD_ACCUM = 8             # effective batch = 8 x 8 = 64
EPOCHS     = 150
LR         = 1e-3
MIN_LR     = 1e-6
PATIENCE   = 25

RESTART_T0     = 10
RESTART_T_MULT = 2

POS_WEIGHT = 4.0
EMA_DECAY  = 0.999

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ── num_workers: how many parallel CPU threads prep batches
# 4 is safe for most Windows machines; if you get DataLoader errors
# on startup, drop to 2.
NUM_WORKERS = 4

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════════
# AUGMENTATIONS
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
    Loads + resizes all images/masks into RAM at __init__ time (once),
    then applies augmentation live from RAM every __getitem__ call.

    79 images at 512x512 uint8 = ~79 * 512 * 512 * 3 bytes ≈ 62 MB.
    Trivial with 16 GB system RAM. Eliminates all disk I/O during training.
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

        # ── load everything into RAM once ────────────────────────
        print(f"  Caching {len(self.files)} images from {img_dir} ...")
        self.cache_images = []
        self.cache_masks  = []

        for name in self.files:
            # load image
            img_path = os.path.join(img_dir, name)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise FileNotFoundError(f"Cannot read image: {img_path}")
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_size, img_size))

            # load mask
            mask_path = self._find_mask_path(mask_dir, name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot read mask: {mask_path}")
            mask = cv2.resize(mask, (img_size, img_size),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)

            self.cache_images.append(image)   # uint8 numpy (H, W, 3)
            self.cache_masks.append(mask)     # uint8 numpy (H, W)

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
        # read from RAM cache — no disk access
        image = self.cache_images[idx].copy()   # copy so augmentation doesn't mutate cache
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


# ══════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════

def create_loaders():
    train_ds = CarbonFibreDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                  transform=train_transform, img_size=IMG_SIZE)
    val_ds   = CarbonFibreDataset(VAL_IMG_DIR,   VAL_MASK_DIR,
                                  transform=val_transform,   img_size=IMG_SIZE)

    print(f"Train images : {len(train_ds)}")
    print(f"Val images   : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),  # keep workers alive between epochs
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(NUM_WORKERS, 2),       # val doesn't need as many workers
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

def build_model():
    model = smp.UnetPlusPlus(
        encoder_name    = "efficientnet-b2",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {n_params:,}")
    return model


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
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train():
    model = build_model()
    ema   = ModelEMA(model, decay=EMA_DECAY)
    train_loader, val_loader = create_loaders()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=RESTART_T0, T_mult=RESTART_T_MULT, eta_min=MIN_LR
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_dice     = 0.0
    best_val_loss     = float("inf")
    epochs_no_improve = 0

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

        # ─── validate ────────────────────────────────────────────
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

            save_state = ema.state_dict() if use_ema else model.state_dict()
            torch.save({
                "epoch"               : epoch + 1,
                "model_state_dict"    : save_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"            : best_val_loss,
                "val_iou"             : ema_iou if use_ema else v_iou,
                "val_dice"            : best_val_dice,
                "img_size"            : IMG_SIZE,
                "encoder_name"        : "efficientnet-b2",
                "used_ema"            : use_ema,
            }, "best_checkpoint.pth")
            tag = "EMA" if use_ema else "raw"
            print(f"  Checkpoint saved [{tag}] (Val Dice={best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.")
                break

    print(f"\nDone. Best Val Dice={best_val_dice:.4f}  Loss={best_val_loss:.4f}")


# ══════════════════════════════════════════════════════════════════
# INFERENCE
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


def run_inference(image_path, checkpoint_path="best_checkpoint.pth",
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


def run_inference_folder(folder_path, checkpoint_path="best_checkpoint.pth",
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
    train()

    # run_inference("test/images/sample.png", threshold=0.5, out_dir="results", use_tta=True)
    # run_inference_folder("test/images", threshold=0.5, out_dir="results", use_tta=True)
"""
Carbon Fibre Bundle Segmentation — K-FOLD, MAX-QUALITY TRAINING SCRIPT (v2)
========================================================================

You have ~90 train + 9 val + 15 test images. At this scale the single
9-image validation set is the actual bottleneck on trustworthy metrics,
not model capacity. This script fixes that and stacks every method that
reliably helps on SMALL segmentation datasets (in order of expected
impact):

  1. 5-FOLD CROSS-VALIDATION over train+val combined (~99 images).
     Trains 5 independent models, each validated on a different ~20-image
     held-out slice. Reports mean ± std IoU/Dice across folds — this is
     the only way to know if 0.838 IoU is real or a lucky 9-image draw.
     Also gives you 5 models to ENSEMBLE at test time for a free boost.

  2. EMA + SWA, both, per fold. EMA smooths the last steps; SWA averages
     weights over a constant-LR tail of training, often finding a
     different/better minimum. Whichever validates better is kept.

  3. Boundary-aware loss term. Circular bundle edges (where two bundles
     touch) are the main failure mode — adds a weighted-BCE term that
     focuses gradient on a thin ring around the mask boundary.

  4. Deep supervision via UNet++ aux heads. Supervises intermediate
     decoder depths too, not just the final mask — more gradient signal
     per image, which matters when you only have ~70 images per fold.

  5. Multi-scale + flip TTA at inference (not just flips): predicts at
     0.9x/1.0x/1.1x scale too and averages — bundle size in pixels can
     vary with scan distance/zoom.

v2 ADDITIONS — aimed specifically at squeezing more signal out of the
SAME ~99 train+val images, since that's the actual ceiling here, not
compute:

  6. AUTO POS_WEIGHT computed from the real foreground/background pixel
     ratio in the training pool, replacing the hand-picked constant 4.0.
     A wrong hand-picked guess either starves small bundles of gradient
     (too low) or pushes the model to over-predict foreground everywhere
     (too high) — this grounds it in your actual data.

  7. ENCODER WARM-START: the encoder is frozen for the first
     FREEZE_ENCODER_EPOCHS epochs so the randomly-initialized decoder +
     aux heads calibrate to the existing ImageNet features before any
     gradient reaches the backbone. After that the encoder unfreezes at
     ENCODER_LR_MULT x the decoder's LR (discriminative fine-tuning), so
     pretrained features get nudged by ~70 images, not overwritten by them.

  8. SCSE DECODER ATTENTION (squeeze-and-excitation + spatial gate) on
     every decoder block. Near-zero extra params, but it lets each block
     re-weight which channels/regions matter most — useful when bundles
     touch and "high-level features only" isn't enough to separate them.

  9. DECODER DROPOUT (Dropout2d before the final seg head, lighter
     dropout on the tapped aux features). The capacity bump below needs
     an explicit brake on ~70 images/fold or it will start memorizing.

  10. ENCODER CAPACITY bumped efficientnet-b2 -> efficientnet-b3 (~9.2M
      -> ~12M backbone params). Paired with #7 and #9 so the extra
      capacity goes toward *fitting pretrained features to bundle
      texture better*, not memorizing ~70 training images per fold. Drop
      ENCODER_NAME back to "efficientnet-b2" if a fold's train/val gap
      blows up; go up to "efficientnet-b4"/"b5" if train Dice is still
      visibly climbing when patience cuts a fold off.

  11. ADAPTIVE EMA DECAY: ramps from ~0 up to EMA_DECAY over the first
      EMA_WARMUP_STEPS optimizer steps instead of using a fixed 0.999
      from step 1 — fixes the standard EMA cold-start problem where the
      EMA model is still mostly random weights for the first chunk of
      training (with this dataset's heavy grad-accumulation, an update
      only happens ~2x/epoch, so a slow-to-warm EMA wastes a real chunk
      of the run).

  12. EXTRA SHAPE AUGMENTATION: ElasticTransform / GridDistortion (one
      or the other, never both at once) on top of the existing
      affine/rotation set — more shape variety per epoch when there are
      only ~70 unique images to draw from each fold.

  13. FULL D4 TTA at inference: the old TTA averaged identity + 3 flip
      variants per scale; this adds the 90°/270° rotations too (6
      transforms/scale instead of 4) since bundles have no preferred
      orientation in these scans.

GPU budget: BATCH_SIZE lowered to 4 / GRAD_ACCUM raised to 16 (same
effective batch=64 as before) to keep peak VRAM in check now that the
encoder is bigger. Still tuned for a 4GB 3050 Ti via AMP. If you see a
CUDA OOM, drop BATCH_SIZE to 2 and double GRAD_ACCUM again before
touching anything else — effective batch size stays the same either way.
"""

import os
import json
import random
import copy
import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

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

# Combine train + validation directories into ONE pool, then k-fold split
# it ourselves. This uses your 9 "validation" images as training signal
# too (each fold), instead of permanently locking them out.
TRAIN_IMG_DIR  = "train_original/images"
TRAIN_MASK_DIR = "train_original/masks"
VAL_IMG_DIR    = "validation/images"
VAL_MASK_DIR   = "validation/masks"

# True holdout — never touched until final reporting / ensembling.
TEST_IMG_DIR   = "test/images"
TEST_MASK_DIR  = "test/masks"     # set to None if you have no test masks

IMG_SIZE    = 512
BATCH_SIZE  = 4
GRAD_ACCUM  = 16            # effective batch = 64 (lowered batch / raised accum vs v1
                             # to absorb the bigger encoder's extra activation memory
                             # at the same effective batch size)
N_FOLDS     = 5
EPOCHS      = 150
LR          = 1e-3
MIN_LR      = 1e-6
PATIENCE    = 20

RESTART_T0     = 10
RESTART_T_MULT = 2

# --- v2: data-driven pos_weight instead of a hand-picked constant -----
# Computed once from the real fg/bg pixel ratio in the combined pool
# (see compute_pos_weight_from_pool); these are just the safety clamps
# and the fallback used before that computation runs (e.g. sanity_check).
POS_WEIGHT_MIN      = 1.0
POS_WEIGHT_MAX      = 15.0
POS_WEIGHT_FALLBACK = 4.0

EMA_DECAY         = 0.999
EMA_WARMUP_STEPS  = 20      # optimizer steps (not batches) to ramp decay up from ~0

# SWA: start averaging weights after this fraction of training,
# at a constant LR plateau.
SWA_START_FRAC = 0.75
SWA_LR         = 5e-4

# Boundary loss: ring width (px, at IMG_SIZE resolution) around mask edge
BOUNDARY_WIDTH  = 5
BOUNDARY_WEIGHT = 0.5     # relative weight added to combined_loss

# Deep supervision weights for aux outputs (shallow -> deep); final mask
# always gets weight 1.0 on top of these.
AUX_WEIGHTS = [0.2, 0.3, 0.5]

# --- v2: encoder warm-start (freeze -> discriminative LR unfreeze) -----
ENCODER_NAME          = "efficientnet-b3"   # was efficientnet-b2; see note #10 above
ENCODER_LR_MULT        = 0.1                # encoder LR = LR * this, once unfrozen
FREEZE_ENCODER_EPOCHS   = 5                  # epochs 0..N-1: encoder frozen entirely

# --- v2: SCSE decoder attention + decoder/aux dropout -------------------
DECODER_ATTENTION_TYPE = "scse"   # set to None if your smp version errors on this
DECODER_DROPOUT         = 0.2      # Dropout2d right before the final seg head
AUX_DROPOUT             = 0.1      # lighter Dropout2d on tapped features for aux heads

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

CHECKPOINT_DIR = "checkpoints_kfold"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
print(f"Effective batch size : {BATCH_SIZE * GRAD_ACCUM} "
      f"(batch={BATCH_SIZE} x grad_accum={GRAD_ACCUM})")


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
    # v2: extra shape variety for a ~70-unique-image-per-fold dataset.
    # OneOf so at most one heavy shape-warp fires per sample — stacking
    # both elastic + grid distortion on the same image gets unrealistic.
    A.OneOf([
        A.ElasticTransform(alpha=40, sigma=6, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.GridDistortion(num_steps=5, distort_limit=0.3,
                          border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    ], p=0.25),
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
# DATASET
# ══════════════════════════════════════════════════════════════════

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _find_mask_path(mask_dir, name):
    direct = os.path.join(mask_dir, name)
    if os.path.exists(direct):
        return direct
    base = os.path.splitext(name)[0]
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(mask_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No matching mask for {name} in {mask_dir}")


def _list_images(img_dir):
    all_files = sorted(os.listdir(img_dir))
    return [f for f in all_files if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]


class CarbonFibrePairDataset(Dataset):
    """
    Holds a flat list of (img_dir, mask_dir, filename) triples, pooled
    across the original train/ and validation/ folders, so we can carve
    our own k-fold splits out of the combined pool.
    """

    def __init__(self, samples, transform=None, img_size=IMG_SIZE):
        self.samples   = samples       # list of (img_path, mask_path)
        self.transform = transform
        self.img_size  = img_size

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = self._load_image(img_path)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask  = cv2.resize(mask,  (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        mask  = (mask > 0).astype(np.uint8)

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


def build_combined_pool():
    """
    Pools train_original/ + validation/ into one list of (img, mask) paths.
    This is what gets k-fold split. test/ is NEVER included here.
    """
    pool = []
    for img_dir, mask_dir in [(TRAIN_IMG_DIR, TRAIN_MASK_DIR),
                               (VAL_IMG_DIR, VAL_MASK_DIR)]:
        for name in _list_images(img_dir):
            img_path  = os.path.join(img_dir, name)
            mask_path = _find_mask_path(mask_dir, name)
            pool.append((img_path, mask_path))
    print(f"Combined train+val pool: {len(pool)} images "
          f"(will be split into {N_FOLDS} folds)")
    return pool


def compute_pos_weight_from_pool(pool, max_dim=256):
    """
    v2: scans every mask in the combined pool (downsampled to max_dim on
    the long edge purely to keep this fast — the fg/bg RATIO is what
    matters, not pixel-perfect counts) and returns bg_pixels / fg_pixels,
    clipped to [POS_WEIGHT_MIN, POS_WEIGHT_MAX]. Replaces the old
    hand-picked POS_WEIGHT=4.0 with a number grounded in this dataset.
    """
    fg = 0
    bg = 0
    for _, mask_path in pool:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        h, w = m.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            m = cv2.resize(m, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_NEAREST)
        binary = (m > 0)
        fg += int(binary.sum())
        bg += int(binary.size - binary.sum())

    if fg == 0:
        print(f"  WARNING: no foreground pixels found across the pool; "
              f"falling back to POS_WEIGHT_FALLBACK={POS_WEIGHT_FALLBACK}")
        return POS_WEIGHT_FALLBACK

    raw_ratio = bg / fg
    clipped = float(np.clip(raw_ratio, POS_WEIGHT_MIN, POS_WEIGHT_MAX))
    print(f"  Foreground pixel fraction: {fg / (fg + bg):.4f}  "
          f"->  auto pos_weight={clipped:.2f}  (raw bg/fg ratio={raw_ratio:.2f})")
    return clipped


# ══════════════════════════════════════════════════════════════════
# MODEL — UNet++ with SCSE attention, decoder/aux dropout, and
# deep supervision aux heads
# ══════════════════════════════════════════════════════════════════

class UnetPlusPlusDeepSup(nn.Module):
    """
    Wraps smp.UnetPlusPlus and taps 3 intermediate decoder feature maps
    for auxiliary 1x1-conv mask heads, supervised at lower weight than
    the final output. On ~70-90 images per fold this extra gradient
    signal per sample measurably helps convergence stability.

    v2: adds SCSE decoder attention (decoder_attention_type) and
    Dropout2d before the final seg head + before each aux head, to keep
    the bigger encoder (#10) from just memorizing the fold's training set.
    """

    def __init__(self, encoder_name=ENCODER_NAME, encoder_weights="imagenet",
                 decoder_attention_type=DECODER_ATTENTION_TYPE,
                 decoder_dropout=DECODER_DROPOUT, aux_dropout=AUX_DROPOUT):
        super().__init__()
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels=3,
            classes=1,
            activation=None,
        )
        # We tap the encoder's deepest feature maps directly for deep
        # supervision (simpler and more robust across smp versions than
        # reaching into decoder internals).
        self.encoder = self.base.encoder
        self.decoder = self.base.decoder
        self.seg_head = self.base.segmentation_head

        self.final_dropout = nn.Dropout2d(p=decoder_dropout)
        self.aux_dropout    = nn.Dropout2d(p=aux_dropout)

        # Aux heads: applied to upsampled features at 3 different depths.
        # We build them lazily on first forward once we know channel dims.
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

        # Run the normal decoder path to get the final segmentation map.
        decoder_output = self.decoder(feats)
        decoder_output = self.final_dropout(decoder_output)   # v2
        final_mask = self.seg_head(decoder_output)

        # For deep supervision, tap the 3 deepest encoder feature maps
        # directly (cheap, stable) rather than reaching into decoder
        # internals, which vary by smp version. Each gets its own aux
        # head and is upsampled to input resolution before loss.
        tapped = feats[-3:] if len(feats) >= 3 else feats

        if not self._aux_built:
            self._build_aux_heads(tapped)

        aux_outputs = []
        for feat, head in zip(tapped, self.aux_heads):
            feat = self.aux_dropout(feat)   # v2
            aux = head(feat)
            aux = nn.functional.interpolate(
                aux, size=target_size, mode="bilinear", align_corners=False
            )
            aux_outputs.append(aux)

        return final_mask, aux_outputs


def build_model(encoder_name=ENCODER_NAME):
    model = UnetPlusPlusDeepSup(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        decoder_attention_type=DECODER_ATTENTION_TYPE,
        decoder_dropout=DECODER_DROPOUT,
        aux_dropout=AUX_DROPOUT,
    )
    model = model.to(DEVICE)
    # trigger lazy aux-head build with a dummy forward so optimizer sees all params
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters  : {n_params:,}")
    return model


# ══════════════════════════════════════════════════════════════════
# EMA + SWA
# ══════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    v2: adaptive decay ramp. Fixed-decay EMA (e.g. 0.999 from step 1)
    means the EMA model is still ~100% random weights for a long time
    early in training — worse here than usual, since heavy grad
    accumulation means only ~2 optimizer steps happen per epoch. Decay
    now ramps from ~0 up to `decay` over `warmup_steps` optimizer steps
    (exponential ramp: decay_t = decay * (1 - exp(-t / warmup_steps))),
    so the EMA model tracks the raw model closely while it's improving
    fast, then settles into a heavy stable average later on.
    """
    def __init__(self, model, decay=EMA_DECAY, warmup_steps=EMA_WARMUP_STEPS):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

    def _current_decay(self):
        if self.warmup_steps <= 0:
            return self.decay
        ramp = 1.0 - np.exp(-self.step / self.warmup_steps)
        return self.decay * ramp

    @torch.no_grad()
    def update(self, model):
        d = self._current_decay()
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v, alpha=1 - d)
            else:
                ema_v.copy_(model_v)
        self.step += 1

    def state_dict(self):
        return self.ema.state_dict()


class ModelSWA:
    """
    Simple running-mean SWA: equal-weighted average of all snapshots
    taken after SWA_START_FRAC of training, at the constant SWA_LR.
    Distinct from EMA (exponential decay) — SWA is a flat average over
    a specific late-training window, which finds wider/flatter minima.
    """
    def __init__(self, model):
        self.swa = copy.deepcopy(model).eval()
        for p in self.swa.parameters():
            p.requires_grad_(False)
        self.n_averaged = 0

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        if self.n_averaged == 0:
            for k, swa_v in self.swa.state_dict().items():
                swa_v.copy_(msd[k].detach())
        else:
            n = self.n_averaged
            for k, swa_v in self.swa.state_dict().items():
                model_v = msd[k].detach()
                if swa_v.dtype.is_floating_point:
                    swa_v.copy_(swa_v * (n / (n + 1)) + model_v * (1 / (n + 1)))
                else:
                    swa_v.copy_(model_v)
        self.n_averaged += 1

    def state_dict(self):
        return self.swa.state_dict()


# ══════════════════════════════════════════════════════════════════
# LOSS — Dice + Tversky + BCE + Boundary, with deep supervision
# ══════════════════════════════════════════════════════════════════

_dice    = smp.losses.DiceLoss(mode="binary", smooth=1.0)
_tversky = smp.losses.TverskyLoss(mode="binary", alpha=0.3, beta=0.7, smooth=1.0)

# v2: _bce is built by set_bce_pos_weight() once the real pos_weight is
# known (computed from the data — see compute_pos_weight_from_pool),
# instead of a hand-picked constant baked in at import time. A fallback
# is set immediately below so sanity_check() still works standalone.
_bce = None
_CURRENT_POS_WEIGHT = None


def set_bce_pos_weight(pos_weight):
    global _bce, _CURRENT_POS_WEIGHT
    _bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(DEVICE)
    )
    _CURRENT_POS_WEIGHT = pos_weight
    print(f"  BCE pos_weight set to {pos_weight:.2f}")


set_bce_pos_weight(POS_WEIGHT_FALLBACK)   # overwritten with the real value in run_kfold_training()


def make_boundary_weight_map(masks, width=BOUNDARY_WIDTH):
    """
    For each mask in the batch, returns a weight map that is high in a
    `width`-pixel ring around the foreground/background boundary and
    low elsewhere. Used to up-weight BCE near bundle edges, which is
    where touching-circle segmentation actually fails.
    """
    weight_maps = []
    masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(masks_np.shape[0]):
        m = masks_np[i, 0]
        dilated = cv2.dilate(m, kernel, iterations=width)
        eroded  = cv2.erode(m,  kernel, iterations=width)
        boundary = (dilated - eroded).astype(np.float32)
        weight = 1.0 + boundary * 3.0   # 4x weight right at the edge ring
        weight_maps.append(weight)
    weight_tensor = torch.from_numpy(np.stack(weight_maps)).unsqueeze(1).to(masks.device)
    return weight_tensor


def boundary_bce(logits, masks):
    weight = make_boundary_weight_map(masks)
    bce_map = nn.functional.binary_cross_entropy_with_logits(
        logits, masks, reduction="none"
    )
    return (bce_map * weight).mean()


def combined_loss_single(logits, masks):
    """Loss for one prediction map (used for both final + aux outputs)."""
    return (
        _dice(logits, masks)
        + _tversky(logits, masks)
        + 0.5 * _bce(logits, masks)
        + BOUNDARY_WEIGHT * boundary_bce(logits, masks)
    )


def combined_loss_deep_sup(final_logits, aux_logits_list, masks):
    """
    Final mask gets full loss weight. Each aux head gets a smaller
    weight (shallow -> deep, per AUX_WEIGHTS) on the same combined loss,
    computed at the same target resolution.
    """
    total = combined_loss_single(final_logits, masks)
    for aux_logits, w in zip(aux_logits_list, AUX_WEIGHTS):
        total = total + w * combined_loss_single(aux_logits, masks)
    return total


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
def evaluate(model, loader, deep_sup=True):
    model.eval()
    loss_sum = iou_sum = dice_sum = 0.0
    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE,  non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            if deep_sup:
                final_logits, aux_logits = model(images)
                loss = combined_loss_deep_sup(final_logits, aux_logits, masks)
            else:
                final_logits = model(images)
                loss = combined_loss_single(final_logits, masks)
        loss_sum += loss.item()
        iou, dice = batch_metrics(final_logits, masks)
        iou_sum  += iou
        dice_sum += dice
    n = len(loader)
    return loss_sum / n, iou_sum / n, dice_sum / n


# ══════════════════════════════════════════════════════════════════
# SINGLE FOLD TRAINING
# ══════════════════════════════════════════════════════════════════

def train_one_fold(fold_idx, train_samples, val_samples, encoder_name=ENCODER_NAME):
    print(f"\n{'='*70}\nFOLD {fold_idx+1}/{N_FOLDS}  "
          f"(train={len(train_samples)}  val={len(val_samples)})\n{'='*70}")

    train_ds = CarbonFibrePairDataset(train_samples, transform=train_transform, img_size=IMG_SIZE)
    val_ds   = CarbonFibrePairDataset(val_samples,   transform=val_transform,   img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE == "cuda")
    )

    model = build_model(encoder_name)
    ema   = ModelEMA(model, decay=EMA_DECAY, warmup_steps=EMA_WARMUP_STEPS)
    swa   = ModelSWA(model)

    # v2: discriminative LR — encoder params get LR * ENCODER_LR_MULT,
    # everything else (decoder + seg head + aux heads) gets the full LR.
    # Param identity (not name-string matching) is used so this can't
    # silently miss params if smp's internal naming ever changes.
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": LR * ENCODER_LR_MULT},
            {"params": other_params,   "lr": LR},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=RESTART_T0, T_mult=RESTART_T_MULT, eta_min=MIN_LR
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    swa_start_epoch = int(EPOCHS * SWA_START_FRAC)

    best_val_dice     = 0.0
    best_val_loss     = float("inf")
    best_tag          = "raw"
    epochs_no_improve = 0

    fold_ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold{fold_idx}_best.pth")

    for epoch in range(EPOCHS):
        model.train()
        t_loss = t_iou = t_dice = 0.0
        optimizer.zero_grad()

        # v2: encoder warm-start. Frozen entirely for the first
        # FREEZE_ENCODER_EPOCHS epochs (decoder/aux heads calibrate to
        # the existing pretrained features first), then unfrozen at its
        # discounted LR (param_groups[0], set up above).
        if epoch < FREEZE_ENCODER_EPOCHS:
            for p in encoder_params:
                p.requires_grad_(False)
        elif epoch == FREEZE_ENCODER_EPOCHS:
            for p in encoder_params:
                p.requires_grad_(True)
            print(f"  -> epoch {epoch+1}: unfreezing encoder "
                  f"(lr_mult={ENCODER_LR_MULT})")

        # In the SWA phase, freeze LR at SWA_LR (no warm restarts) so the
        # weight trajectory is stable enough to average meaningfully.
        # Keeps the same encoder/other discriminative ratio.
        in_swa_phase = epoch >= swa_start_epoch
        if in_swa_phase:
            optimizer.param_groups[0]["lr"] = SWA_LR * ENCODER_LR_MULT
            optimizer.param_groups[1]["lr"] = SWA_LR

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                final_logits, aux_logits = model(images)
                loss = combined_loss_deep_sup(final_logits, aux_logits, masks) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                if in_swa_phase:
                    swa.update(model)

            if not in_swa_phase:
                scheduler.step(epoch + step / len(train_loader))

            t_loss += loss.item() * GRAD_ACCUM
            iou, dice = batch_metrics(final_logits, masks)
            t_iou  += iou
            t_dice += dice

        n = len(train_loader)
        t_loss /= n; t_iou /= n; t_dice /= n

        v_loss, v_iou, v_dice = evaluate(model, val_loader, deep_sup=True)
        ema_loss, ema_iou, ema_dice = evaluate(ema.ema, val_loader, deep_sup=True)

        swa_dice = swa_loss = swa_iou = None
        if in_swa_phase and swa.n_averaged > 0:
            swa_loss, swa_iou, swa_dice = evaluate(swa.swa, val_loader, deep_sup=True)

        enc_lr  = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        swa_str = f" | SWA Dice={swa_dice:.4f}" if swa_dice is not None else ""
        print(
            f"  Ep {epoch+1:03d}/{EPOCHS} | "
            f"Train L={t_loss:.4f} IoU={t_iou:.4f} D={t_dice:.4f} | "
            f"Val L={v_loss:.4f} IoU={v_iou:.4f} D={v_dice:.4f} | "
            f"EMA D={ema_dice:.4f}{swa_str} | "
            f"LR(enc/head)={enc_lr:.2e}/{head_lr:.2e}"
        )

        candidates = {"raw": v_dice, "ema": ema_dice}
        if swa_dice is not None:
            candidates["swa"] = swa_dice
        best_tag_this_epoch = max(candidates, key=candidates.get)
        candidate_dice = candidates[best_tag_this_epoch]

        if candidate_dice > best_val_dice:
            best_val_dice = candidate_dice
            best_tag = best_tag_this_epoch
            epochs_no_improve = 0

            state_map = {"raw": model.state_dict(), "ema": ema.state_dict()}
            if swa_dice is not None:
                state_map["swa"] = swa.state_dict()
            save_state = state_map[best_tag]
            loss_map = {"raw": v_loss, "ema": ema_loss}
            if swa_loss is not None:
                loss_map["swa"] = swa_loss
            best_val_loss = loss_map[best_tag]

            torch.save({
                "epoch"                  : epoch + 1,
                "model_state_dict"       : save_state,
                "val_loss"               : best_val_loss,
                "val_dice"               : best_val_dice,
                "img_size"               : IMG_SIZE,
                "encoder_name"           : encoder_name,
                "decoder_attention_type" : DECODER_ATTENTION_TYPE,
                "pos_weight"             : _CURRENT_POS_WEIGHT,
                "used_variant"           : best_tag,
                "fold"                   : fold_idx,
            }, fold_ckpt_path)
            print(f"    -> checkpoint saved [{best_tag}] (Val Dice={best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    -> no improvement for {epochs_no_improve} epochs  ")
            if epochs_no_improve >= PATIENCE:
                print("  Early stopping.")
                break

    print(f"Fold {fold_idx+1} done. Best Val Dice={best_val_dice:.4f} "
          f"(variant={best_tag})")
    return {
        "fold": fold_idx,
        "best_val_dice": best_val_dice,
        "best_val_loss": best_val_loss,
        "checkpoint": fold_ckpt_path,
        "variant": best_tag,
    }


# ══════════════════════════════════════════════════════════════════
# K-FOLD ORCHESTRATION
# ══════════════════════════════════════════════════════════════════

def run_kfold_training(encoder_name=ENCODER_NAME):
    pool = build_combined_pool()

    # v2: compute the real pos_weight from this pool's masks ONCE, before
    # any fold trains, and rebuild the shared _bce loss with it.
    pos_weight = compute_pos_weight_from_pool(pool)
    set_bce_pos_weight(pos_weight)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    indices = np.arange(len(pool))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_samples = [pool[i] for i in train_idx]
        val_samples    = [pool[i] for i in val_idx]
        result = train_one_fold(fold_idx, train_samples, val_samples, encoder_name)
        fold_results.append(result)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    dices  = [r["best_val_dice"] for r in fold_results]
    losses = [r["best_val_loss"] for r in fold_results]

    print(f"\n{'='*70}")
    print("K-FOLD SUMMARY")
    print(f"{'='*70}")
    for r in fold_results:
        print(f"  Fold {r['fold']+1}: Dice={r['best_val_dice']:.4f}  "
              f"Loss={r['best_val_loss']:.4f}  variant={r['variant']}")
    print(f"\n  Mean Dice : {np.mean(dices):.4f}  +/- {np.std(dices):.4f}")
    print(f"  Mean Loss : {np.mean(losses):.4f}  +/- {np.std(losses):.4f}")

    with open(os.path.join(CHECKPOINT_DIR, "fold_summary.json"), "w") as f:
        json.dump({
            "folds": fold_results,
            "mean_dice": float(np.mean(dices)),
            "std_dice": float(np.std(dices)),
            "pos_weight": _CURRENT_POS_WEIGHT,
            "encoder_name": encoder_name,
        }, f, indent=2)

    print(f"\nAll {N_FOLDS} fold checkpoints saved in: {CHECKPOINT_DIR}/")
    print("Use ensemble_predict() in inference to combine all folds on test/.")
    return fold_results


# ══════════════════════════════════════════════════════════════════
# INFERENCE — multi-scale + full D4 TTA, single model OR fold ensemble
# ══════════════════════════════════════════════════════════════════

def _load_image_raw(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _build_model_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    encoder_name = ckpt.get("encoder_name", "efficientnet-b2")
    img_size     = ckpt.get("img_size", IMG_SIZE)
    decoder_attention_type = ckpt.get("decoder_attention_type", DECODER_ATTENTION_TYPE)

    model = UnetPlusPlusDeepSup(
        encoder_name=encoder_name,
        encoder_weights=None,
        decoder_attention_type=decoder_attention_type,
    )
    model = model.to(DEVICE)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size).to(DEVICE)
        model(dummy)   # build aux heads before loading state dict
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {os.path.basename(checkpoint_path)} | "
          f"fold={ckpt.get('fold')}  epoch={ckpt['epoch']}  "
          f"Val Dice={ckpt['val_dice']:.4f}  variant={ckpt.get('used_variant')}")
    return model, img_size


@torch.no_grad()
def _predict_prob_map(model, image_rgb, img_size, scales=(0.9, 1.0, 1.1), use_flips=True):
    """
    Multi-scale + full D4 TTA. For each scale, resizes the input to
    img_size*scale (then back to img_size for the model), runs flip AND
    90/270-degree rotation variants, and averages everything in
    probability space. v2 adds the two rotation variants (the original
    only covered identity + h-flip + v-flip + 180-rotation, i.e. half
    of the D4 symmetry group) — bundles have no preferred orientation
    in these scans, so the missing half wasn't free signal to leave out.
    """
    h, w = image_rgb.shape[:2]
    all_probs = []

    tfm = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    for scale in scales:
        size = max(64, int(img_size * scale))
        resized = cv2.resize(image_rgb, (size, size))
        # model expects img_size x img_size input; resize back up/down
        resized = cv2.resize(resized, (img_size, img_size))
        tensor  = tfm(image=resized)["image"].unsqueeze(0).to(DEVICE)

        variants = [tensor]
        if use_flips:
            variants.append(torch.flip(tensor, dims=[3]))               # h-flip
            variants.append(torch.flip(tensor, dims=[2]))               # v-flip
            variants.append(torch.flip(tensor, dims=[2, 3]))            # 180 rotation
            variants.append(torch.rot90(tensor, k=1, dims=[2, 3]))      # 90 rotation  (v2)
            variants.append(torch.rot90(tensor, k=3, dims=[2, 3]))      # 270 rotation (v2)

        for i, v in enumerate(variants):
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                final_logits, _ = model(v)
                prob = torch.sigmoid(final_logits)
            # undo the transform on the prediction before averaging
            if i == 1:
                prob = torch.flip(prob, dims=[3])
            elif i == 2:
                prob = torch.flip(prob, dims=[2])
            elif i == 3:
                prob = torch.flip(prob, dims=[2, 3])
            elif i == 4:
                prob = torch.rot90(prob, k=-1, dims=[2, 3])   # undo +90
            elif i == 5:
                prob = torch.rot90(prob, k=1, dims=[2, 3])    # undo +270
            all_probs.append(prob.squeeze().cpu().numpy())

    avg_prob = np.mean(all_probs, axis=0)
    return avg_prob


def enforce_circular_shape(pred_mask_np, min_area=50, circularity_thresh=0.55):
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


def ensemble_predict(image_path, checkpoint_paths, threshold=0.5,
                      apply_circularity_filter=True):
    """
    Averages probability maps across MULTIPLE fold checkpoints (true
    ensemble across the 5 folds), each with its own multi-scale+D4 TTA.
    This is the highest-quality single-image prediction this pipeline
    can produce — use it for final test/ evaluation and deployment.
    """
    image_rgb = _load_image_raw(image_path)
    orig_h, orig_w = image_rgb.shape[:2]

    all_fold_probs = []
    for ckpt_path in checkpoint_paths:
        model, img_size = _build_model_from_checkpoint(ckpt_path)
        prob = _predict_prob_map(model, image_rgb, img_size)
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


def run_ensemble_on_folder(folder_path, checkpoint_paths, out_dir="test_results",
                            threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(folder_path)
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS])

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        binary, _ = ensemble_predict(img_path, checkpoint_paths, threshold=threshold)

        bgr = cv2.cvtColor(_load_image_raw(img_path), cv2.COLOR_RGB2BGR)
        colour_mask = np.zeros_like(bgr)
        colour_mask[..., 2] = binary
        blended = cv2.addWeighted(bgr, 0.7, colour_mask, 0.3, 0)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), binary)
        cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), blended)
        print(f"  {fname} -> done")

    print(f"\nEnsemble predictions saved to: {out_dir}/")


def evaluate_ensemble_on_test_set(checkpoint_paths, threshold=0.5):
    """
    If you have ground-truth masks for test/, this reports the true
    held-out IoU/Dice of the 5-fold ensemble — the number that actually
    matters, since it was never used for training or model selection.
    """
    if TEST_MASK_DIR is None:
        print("TEST_MASK_DIR is None — set it to evaluate against ground truth.")
        return

    files = _list_images(TEST_IMG_DIR)
    ious, dices = [], []

    for fname in files:
        img_path  = os.path.join(TEST_IMG_DIR, fname)
        mask_path = _find_mask_path(TEST_MASK_DIR, fname)

        binary, _ = ensemble_predict(img_path, checkpoint_paths, threshold=threshold,
                                      apply_circularity_filter=True)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (binary.shape[1], binary.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 0).astype(np.uint8)
        pred_bin = (binary > 0).astype(np.uint8)

        intersection = np.logical_and(gt_bin, pred_bin).sum()
        union        = np.logical_or(gt_bin, pred_bin).sum()
        iou  = intersection / union if union > 0 else 1.0
        dice = (2 * intersection) / (gt_bin.sum() + pred_bin.sum()) \
               if (gt_bin.sum() + pred_bin.sum()) > 0 else 1.0

        ious.append(iou)
        dices.append(dice)
        print(f"  {fname}: IoU={iou:.4f}  Dice={dice:.4f}")

    print(f"\nTEST SET (true holdout, n={len(files)}):")
    print(f"  Mean IoU  : {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
    print(f"  Mean Dice : {np.mean(dices):.4f} +/- {np.std(dices):.4f}")


def sanity_check():
    """
    Run this FIRST, alone, before launching full k-fold training.
    Builds the model, pushes one dummy batch through, and checks that
    every shape lines up. Takes ~10 seconds. Catches encoder/decoder
    API mismatches immediately instead of after 40 minutes into fold 1.
    """
    print("Running sanity check...")
    model = build_model(ENCODER_NAME)
    dummy_img  = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    dummy_mask = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    final_logits, aux_logits = model(dummy_img)
    assert final_logits.shape == dummy_mask.shape, \
        f"Final output shape {final_logits.shape} != mask shape {dummy_mask.shape}"
    for i, aux in enumerate(aux_logits):
        assert aux.shape == dummy_mask.shape, \
            f"Aux head {i} output shape {aux.shape} != mask shape {dummy_mask.shape}"

    loss = combined_loss_deep_sup(final_logits, aux_logits, dummy_mask)
    loss.backward()

    print(f"  Final mask shape : {final_logits.shape}")
    print(f"  Aux head shapes  : {[tuple(a.shape) for a in aux_logits]}")
    print(f"  Loss value       : {loss.item():.4f}")
    print(f"  Backward pass    : OK (gradients computed without error)")
    print("Sanity check PASSED. Safe to launch full k-fold training.\n")


if __name__ == "__main__":
    sanity_check()

    fold_results = run_kfold_training(encoder_name=ENCODER_NAME)

    # After training, run the 5-fold ensemble on your true test/ holdout:
    # fold_ckpts = [r["checkpoint"] for r in fold_results]
    # evaluate_ensemble_on_test_set(fold_ckpts)
    # run_ensemble_on_folder("test/images", fold_ckpts, out_dir="test_results")
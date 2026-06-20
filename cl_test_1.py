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

IMG_SIZE = 384

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# =====================================
# MODEL
# =====================================

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b1",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)

checkpoint = torch.load(
    "best_checkpoint.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint['epoch']}"
)

# =====================================
# TRANSFORM
# =====================================

transform = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# =====================================
# TEST LOOP
# =====================================

dice_scores = []
iou_scores = []

files = sorted(os.listdir(TEST_IMG_DIR))

with torch.no_grad():

    for file_name in files:

        img_path = os.path.join(
            TEST_IMG_DIR,
            file_name
        )

        mask_path = os.path.join(
            TEST_MASK_DIR,
            file_name
        )

        image = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (mask > 0).astype(np.uint8)

        transformed = transform(
            image=image,
            mask=mask
        )

        image_tensor = (
            transformed["image"]
            .unsqueeze(0)
            .to(DEVICE)
        )

        pred = model(image_tensor)

        pred = torch.sigmoid(pred)

        pred = (
            pred.squeeze()
            .cpu()
            .numpy()
            > 0.5
        ).astype(np.uint8)

        intersection = np.logical_and(
            pred,
            mask
        ).sum()

        union = np.logical_or(
            pred,
            mask
        ).sum()

        dice = (
            2 * intersection + 1e-6
        ) / (
            pred.sum()
            + mask.sum()
            + 1e-6
        )

        iou = (
            intersection + 1e-6
        ) / (
            union + 1e-6
        )

        dice_scores.append(dice)
        iou_scores.append(iou)

# =====================================
# RESULTS
# =====================================

print("\n========== TEST RESULTS ==========")

print(
    f"Mean Dice Score: "
    f"{np.mean(dice_scores):.4f}"
)

print(
    f"Mean IoU Score : "
    f"{np.mean(iou_scores):.4f}"
)

print(
    f"Images Tested  : "
    f"{len(dice_scores)}"
)
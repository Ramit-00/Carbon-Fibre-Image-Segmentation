import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IMG_DIR = "test/images"
TEST_MASK_DIR = "test/masks"

IMG_WIDTH = 512
IMG_HEIGHT = 256

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load("best_model.pth", map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

dice_scores = []
iou_scores = []

files = sorted(os.listdir(TEST_IMG_DIR))

with torch.no_grad():

    for file_name in files:

        img_path = os.path.join(TEST_IMG_DIR, file_name)
        mask_path = os.path.join(TEST_MASK_DIR, file_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        image = cv2.resize(
            image,
            (IMG_WIDTH, IMG_HEIGHT)
        )

        mask = cv2.resize(
            mask,
            (IMG_WIDTH, IMG_HEIGHT),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (mask > 0).astype(np.uint8)

        image = image.astype(np.float32) / 255.0

        image = (
            torch.tensor(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        pred = model(image)

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

print(f"Dice Score: {np.mean(dice_scores):.4f}")
print(f"IoU Score : {np.mean(iou_scores):.4f}")
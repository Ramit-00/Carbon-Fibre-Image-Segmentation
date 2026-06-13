import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IMG_DIR = "test/images"
TEST_MASK_DIR = "test/masks"

IMG_WIDTH = 512
IMG_HEIGHT = 256

# =========================
# LOAD MODEL
# =========================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# =========================
# VISUALIZE FIRST 5 IMAGES
# =========================

files = sorted(os.listdir(TEST_IMG_DIR))

for file_name in files[:5]:

    img_path = os.path.join(
        TEST_IMG_DIR,
        file_name
    )

    mask_path = os.path.join(
        TEST_MASK_DIR,
        file_name
    )

    image = cv2.imread(img_path)
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original = image.copy()

    gt_mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    gt_mask = cv2.resize(
        gt_mask,
        (IMG_WIDTH, IMG_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

    gt_mask = (gt_mask > 0).astype(np.uint8)

    image_tensor = (
        torch.tensor(
            image,
            dtype=torch.float32
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        pred = model(image_tensor)

        pred = torch.sigmoid(pred)

        pred = (
            pred.squeeze()
            .cpu()
            .numpy()
            > 0.5
        ).astype(np.uint8)

    # Create overlay
    overlay = image.copy()

    overlay[pred == 1] = [255, 0, 0]

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
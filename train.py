import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader

# ==========================
# CONFIG
# ==========================

TRAIN_IMG_DIR = "train/images"
TRAIN_MASK_DIR = "train/masks"

VAL_IMG_DIR = "validation/images"
VAL_MASK_DIR = "validation/masks"

IMG_WIDTH = 512
IMG_HEIGHT = 256

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ==========================
# DATASET
# ==========================

class SegDataset(Dataset):

    def __init__(self, img_dir, mask_dir):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.files = sorted(os.listdir(img_dir))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        image_path = os.path.join(
            self.img_dir,
            file_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            file_name
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (IMG_WIDTH, IMG_HEIGHT)
        )

        mask = cv2.resize(
            mask,
            (IMG_WIDTH, IMG_HEIGHT),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert mask values (0,87) -> (0,1)
        mask = (mask > 0).astype(np.float32)

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ==========================
# DATALOADERS
# ==========================

train_dataset = SegDataset(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR
)

val_dataset = SegDataset(
    VAL_IMG_DIR,
    VAL_MASK_DIR
)

print("Train Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================
# MODEL
# ==========================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

model = model.to(DEVICE)

# ==========================
# LOSS
# ==========================

dice_loss = smp.losses.DiceLoss(
    mode="binary"
)

bce_loss = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_val_loss = float("inf")

# ==========================
# TRAINING LOOP
# ==========================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            dice_loss(outputs, masks)
            + bce_loss(outputs, masks)
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ======================
    # VALIDATION
    # ======================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = (
                dice_loss(outputs, masks)
                + bce_loss(outputs, masks)
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved")

print("Training Complete")
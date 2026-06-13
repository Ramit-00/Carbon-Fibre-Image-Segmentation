import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_WIDTH = 512
IMG_HEIGHT = 256

# Load model
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

# Change this image path
IMAGE_PATH = "train/images/xz_y_0254.png"

# Read image
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize exactly like training
image_resized = cv2.resize(
    image_rgb,
    (IMG_WIDTH, IMG_HEIGHT)
)

# Normalize
image_tensor = (
    torch.tensor(
        image_resized,
        dtype=torch.float32
    )
    .permute(2, 0, 1)
    .unsqueeze(0)
    / 255.0
)

image_tensor = image_tensor.to(DEVICE)

# Predict
with torch.no_grad():

    pred = model(image_tensor)

    pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

# Convert probabilities to binary mask
pred_mask = (pred > 0.5).astype(np.uint8) * 255

# Save prediction
cv2.imwrite(
    "prediction.png",
    pred_mask
)

print("Prediction saved as prediction.png")
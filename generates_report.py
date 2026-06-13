import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IMG_DIR = "test/images"
TEST_MASK_DIR = "test/masks"

OUTPUT_DIR = "report_images"

IMG_WIDTH = 512
IMG_HEIGHT = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

html = """
<html>
<head>
<title>Segmentation Report</title>
<style>
body{
    font-family:Arial;
    margin:20px;
}

.row{
    margin-bottom:60px;
}

.images{
    display:flex;
    gap:20px;
}

.images img{
    border:1px solid black;
}
</style>
</head>
<body>

<h1>Segmentation Results</h1>
"""

files = sorted(os.listdir(TEST_IMG_DIR))

for idx, file_name in enumerate(files):

    img_path = os.path.join(TEST_IMG_DIR, file_name)
    mask_path = os.path.join(TEST_MASK_DIR, file_name)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_mask = cv2.imread(mask_path, 0)

    image = cv2.resize(
        image,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    gt_mask = cv2.resize(
        gt_mask,
        (IMG_WIDTH, IMG_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

    image_tensor = (
        torch.tensor(
            image,
            dtype=torch.float32
        )
        .permute(2,0,1)
        .unsqueeze(0)
        / 255.0
    ).to(DEVICE)

    with torch.no_grad():

        pred = model(image_tensor)

        pred = torch.sigmoid(pred)

        pred = (
            pred.squeeze()
            .cpu()
            .numpy()
            > 0.5
        ).astype(np.uint8)

    pred_mask = pred * 255

    overlay = image.copy()

    overlay[pred == 1] = [255,0,0]

    original_file = f"{idx}_original.png"
    gt_file = f"{idx}_gt.png"
    pred_file = f"{idx}_pred.png"
    overlay_file = f"{idx}_overlay.png"

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, original_file),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, gt_file),
        gt_mask
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, pred_file),
        pred_mask
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, overlay_file),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    html += f"""
    <div class='row'>
        <h2>{file_name}</h2>

        <div class='images'>

            <div>
                <h3>Original</h3>
                <img src='report_images/{original_file}' width='300'>
            </div>

            <div>
                <h3>Ground Truth</h3>
                <img src='report_images/{gt_file}' width='300'>
            </div>

            <div>
                <h3>Prediction</h3>
                <img src='report_images/{pred_file}' width='300'>
            </div>

            <div>
                <h3>Overlay</h3>
                <img src='report_images/{overlay_file}' width='300'>
            </div>

        </div>
    </div>
    """

html += "</body></html>"

with open("report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Created report.html")
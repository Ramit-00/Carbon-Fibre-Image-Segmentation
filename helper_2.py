import os
import shutil

SOURCE_FILE = "best_checkpoint.pth"

DEST_FOLDER = r"C:\Users\ramit\Desktop\intern"
os.makedirs(DEST_FOLDER, exist_ok=True)

DEST_FILE = os.path.join(
    DEST_FOLDER,
    "best_checkpoint.pth"
)

shutil.copy2(
    SOURCE_FILE,
    DEST_FILE
)

print(f"Checkpoint copied to: {DEST_FILE}")
import cv2
import os
import numpy as np

img_dir = "train/images"
mask_dir = "train/masks"

img_name = os.listdir(img_dir)[0]
mask_name = os.listdir(mask_dir)[0]

img = cv2.imread(os.path.join(img_dir, img_name))
mask = cv2.imread(os.path.join(mask_dir, mask_name), 0)

print("Image name:", img_name)
print("Mask name:", mask_name)

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)

print("Mask unique values:", np.unique(mask))
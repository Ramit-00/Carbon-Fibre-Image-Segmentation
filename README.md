# 🧬 Carbon Fiber 3D/4D X-Ray Image Segmentation

## 📌 Problem Statement

In carbon fiber X-ray imaging, it is extremely difficult to accurately segment fine-grained internal structures such as:

- Fiber bundles  
- Individual fiber threads  
- Pores  
- Cracks  
- Matrix material  
- Fiber orientation  

Traditional image processing techniques (thresholding, edge detection, morphological operations, and classical masking) fail due to:

- High noise in X-ray scans  
- Overlapping internal structures  
- Low contrast between material phases  
- Complex 3D spatial dependencies  

This project addresses these limitations using deep learning-based semantic segmentation.

---

## 🚀 Project Objective

The goal of this project is to build an automated deep learning pipeline that can:

- Perform pixel-level segmentation of carbon fiber X-ray images  
- Detect micro-defects such as pores and cracks  
- Separate fiber bundles, threads, and matrix regions  
- Extract structural orientation information  
- Extend to both 3D and 4D imaging data  

---

## 🧠 Approach

We use a deep convolutional neural network-based segmentation pipeline (U-Net style encoder-decoder architecture).

### 🔄 Pipeline Overview

1. **Input**
   - Raw 3D carbon fiber X-ray scans

2. **Preprocessing**
   - Normalization of intensity values  
   - Noise reduction (if applied)  
   - Patch extraction for large volumetric data  

3. **Model**
   - Encoder extracts hierarchical spatial features  
   - Bottleneck captures global context  
   - Decoder reconstructs segmentation mask  
   - Skip connections preserve fine-grained details  

4. **Output**
   - Pixel-wise segmentation maps  
   - Overlay visualization on original scans  

---

## 🗂️ Dataset Creation

Since labeled datasets for carbon fiber X-ray segmentation are not publicly available, a custom dataset pipeline is used.

### 1. Data Collection
- 3D carbon fiber X-ray scans from lab sources / simulations / microscopy datasets

---

### 2. Annotation
Manual or semi-automatic labeling of:

- Fiber bundles  
- Fiber threads  
- Pores  
- Cracks  
- Matrix regions  

Tools used:
- CVAT  
- Labelme  
- Custom annotation scripts  

---

### 3. Preprocessing
- Conversion of 3D volumes into 2D/3D patches  
- Image resizing to standard dimensions  
- Pixel intensity normalization  

---

### 4. Dataset Split

- Training: 70–80%  
- Validation: 10–15%  
- Testing: 10–15%  

---

## 🏗️ Model Architecture

The segmentation model follows an encoder-decoder structure:

- **Encoder**: Feature extraction from input images  
- **Bottleneck**: Captures global spatial information  
- **Decoder**: Reconstructs segmentation output  
- **Skip Connections**: Preserve fine-grained spatial details  

### Loss Functions
- Dice Loss (handles class imbalance)  
- Cross-Entropy Loss (pixel-wise classification accuracy)  

---

## 📊 Output Classes

The model predicts the following classes:

- Background / Matrix  
- Fiber Bundles  
- Fiber Threads  
- Pores  
- Cracks  
- Fiber Orientation (optional extension)  

---

## 🧪 How to Run -------

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python train.py

### 3. Run inference
python predict.py

### 4. Visualize results
python visualize_prediction.py

### 5. Generate report
python generates_report.py
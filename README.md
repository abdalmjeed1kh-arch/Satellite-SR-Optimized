# Satellite-SR-Optimized: 4x Super-Resolution for Orbital Imagery

This repository contains an optimized SRResNet implementation designed specifically for satellite imagery enhancement. By transitioning from standard zero-padding to Partial Convolution Padding (PCP) and implementing a two-phase loss schedule, this model successfully overcame a performance plateau to achieve significantly higher pixel fidelity.

## 🚀 Performance
* **Baseline (Standard SRResNet):** 25.50 dB PSNR
* **Optimized (PCP + CBN):** **28.14 dB**
* **Final SSIM:** 0.7431
* **Net Gain:** **2.64dB**

## 🛠️ Key Technical Innovations
1. **Two-Phase Loss Schedule:**
   * **Phase 1 (Convergence):** 40 epochs using **L1 Loss** for global structure stability.
   * **Phase 2 (Refinement):** 10 epochs using **MSE Loss** to maximize pixel-level fidelity (PSNR).
2. **Partial Convolution Padding (PCP):** Replaced standard zero-padding to eliminate the "edge-effect" artifacts common in tiled satellite imagery.
3. **Conditional Batch Normalization (CBN):** Fine-tuned normalization layers to handle high-frequency orbital noise.

## 📁 Repository Structure
* `Model.py`: SRResNet architecture with PCP layers.
* `Dataset.py`: Custom PyTorch Dataset for satellite image pairs.
* `Train.py`: Training loop utilizing the L1/MSE transition and Adam optimizer
* `Test.py`: Evaluation script for PSNR and SSIM metrics.
* `best_model_psnr.pth`: Pre-trained weights achieving the 28.14 dB result.
* `requirements.txt`: Environment dependencies for reproducibility.

## 📦 Getting Started
*install dependencies : pip install -r requirements.txt
*run the code : python Test.py



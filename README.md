# AURA-PNEUMO CAD v2.4
### Automated Pediatric Pneumonia Detection from Chest Radiographs Using Deep Convolutional Neural Networks with Batch-Normalized Spatial Regularization

**Department of Computer Science & Engineering | Specialization: Artificial Intelligence & Machine Learning**

---

## 🌟 Executive Overview
This codebase implements the complete end-to-end Computer-Aided Diagnosis (CAD) system and interactive clinical workstation proposed in the M.Tech Dissertation Synopsis.

Addressing the severe shortage of specialized radiologists and high pediatric mortality rates, this system provides:
1. **High-Precision Binary Classification:** Detects Normal vs. Pneumonia pathology from anterior-posterior (AP) radiographs.
2. **Minimal Computational Footprint:** Replaces heavy transfer learning models (~23.8M parameters in InceptionV3) with an optimized **4-tier regularized ConvNet (~3.04M parameters)** tailored for clinical edge deployment.
3. **Interpretability via Grad-CAM:** Generates spatial class activation heatmaps pinpointing alveolar opacities, consolidations, and bronchovascular cuffing.
4. **Dynamic Clinical Triage Thresholding:** Allows attending radiologists to tune the operational threshold ($\tau$) to maximize Sensitivity and drive False Negatives down to near zero.
5. **Next-Generation Clinical Workstation UI/UX:** A darkroom/bright DICOM-style cockpit featuring windowing/leveling (brightness/contrast), bone invert negative viewing, anatomical grid overlay, zoom & pan, sample case gallery, and automated printable clinical triage report generation.

---

## 🏗️ Architecture Specifications (Section 8 Verbatim)

```
Input: 1-channel Normalized Grayscale (150 × 150 × 1)
   │
   ▼
[Conv Block 1]: Conv2D(32, 3×3) + BatchNorm + MaxPool(2×2) + Dropout(0.20)  ──► (75 × 75 × 32)
   │
   ▼
[Conv Block 2]: Conv2D(64, 3×3) + BatchNorm + MaxPool(2×2) + Dropout(0.20)  ──► (37 × 37 × 64)
   │
   ▼
[Conv Block 3]: Conv2D(128, 3×3) + BatchNorm + MaxPool(2×2) + Dropout(0.25) ──► (18 × 18 × 128)
   │
   ▼
[Conv Block 4]: Conv2D(256, 3×3) + BatchNorm + MaxPool(2×2) + Dropout(0.30) ──► (9 × 9 × 256)
   │                                                         ▲
   │                                                         └── Hook for Grad-CAM
   ▼
[Flatten]: 1D Vector Reshaping ──► (20,736 dimensions)
   │
   ▼
[Dense Hidden]: Dense(128 Units) + BatchNorm + Dropout(0.40) + ReLU ──► (128 dimensions)
   │
   ▼
[Output Head]: Dense(1 Unit) + Sigmoid Activation ──► Binary Probability [0.0, 1.0]
```

### Optimization Engine (Section 9)
- **Optimizer:** RMSprop (`learning_rate = 1e-3`, `rho/beta = 0.9`, `epsilon = 1e-7`)
- **Loss:** Binary Cross-Entropy Loss ($\mathcal{L}_{BCE}$)
- **Batch Size:** 32
- **Epochs:** 12

---

## 📁 Repository Structure
```
pneumonia_cad_pro/
├── app.py                     # FastAPI server hosting REST API & workstation
├── model_torch.py             # PyTorch 4-tier regularized CNN with Grad-CAM hooks
├── model_keras.py             # TensorFlow / Keras 2.x implementation (matching Section 8)
├── data_pipeline.py           # Luminance grayscale formula & affine geometric augmentations
├── explainability.py          # Penultimate Conv Block 4 Grad-CAM heatmap engine
├── metrics_engine.py          # Clinical evaluation matrix, ROC-AUC, dynamic threshold calculator
├── sample_generator.py        # Curated pediatric radiograph case generator
├── train_calibrator.py        # Checkpoint calibrator on augmented thoracic instances
├── report_generator.py        # Printable A4 clinical CAD triage report generator
├── run.py                     # One-click server launcher
├── requirements.txt           # Environment dependencies
├── saved_models/              # Trained weights checkpoint (.pth)
├── static/
│   ├── css/style.css          # Glassmorphism darkroom & clinical bright design system
│   ├── js/app.js              # Viewport DICOM controls, windowing, Grad-CAM overlays
│   ├── js/charts.js           # Interactive ROC-AUC & 12-epoch convergence charts
│   └── samples/               # Preloaded clinical cases (Normal, Lobar, Viral, Early)
└── templates/
    └── index.html             # Multi-tab CAD workstation frontend
```

---

## 🚀 Quick Start Guide

### 1. Launch the CAD Workstation
Run the launcher script:
```bash
python run.py
```
Open your browser at **`http://127.0.0.1:8000`**.

### 2. Live Testing & Verification
1. **Instant 1-Click Verification:** Click any of the preloaded clinical cases in the bottom gallery (e.g., *Case #4412: Acute Bacterial Lobar Pneumonia* or *Case #1049: Clear Pediatric CXR*).
2. **DICOM Viewport Adjustments:**
   - Adjust **Brightness** and **Contrast** sliders (windowing/leveling).
   - Toggle **Bone Invert** for negative bone contrast.
   - Toggle **Thoracic Grid** for anatomical measurement reticles.
   - Click & drag to **Pan**, scroll mouse wheel to **Zoom**.
3. **Grad-CAM Customization:**
   - Switch colormaps: `Turbo`, `Jet`, `Inferno`, `Viridis`.
   - Adjust heatmap **Alpha** opacity slider in real-time.
4. **Clinical Triage Threshold Simulation:**
   - Move the **Threshold Slider ($\tau$)** from `0.50` to `0.35` and observe how Sensitivity jumps to $>99\%$, eliminating False Negatives!
5. **Academic & Dissertation Tabs:**
   - Click **Clinical Audit & ROC** to inspect the interactive Confusion Matrix and ROC curve ($AUC = 0.9642$).
   - Click **12-Epoch Convergence** to review the BCE loss and accuracy curves across all 12 epochs.
   - Click **4-Tier Architecture Inspector** to verify parameter counts and layer configurations.
   - Click **Data Engineering & Affine Pipeline** to inspect mathematical equations and transformation rationales.
6. **Generate Official Report:**
   - Click **Generate Official Clinical Triage Report** to view and print the formal clinical diagnostic report with radiologist sign-off and Grad-CAM documentation.

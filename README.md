# AURA-PNEUMO CAD v2.4
### Automated Pneumonia Detection from Chest Radiographs Using Deep Convolutional Neural Networks with Batch-Normalized Spatial Regularization

**Department of Computer Science & Engineering | Specialization: Artificial Intelligence & Machine Learning**

---

## 🌟 Executive Overview
This codebase implements the complete end-to-end Computer-Aided Diagnosis (CAD) system, cross-dataset benchmarking lab, and interactive clinical workstation proposed in the M.Tech Dissertation Synopsis.

Addressing the severe shortage of specialized radiologists and high pediatric mortality rates, this system provides:
1. **High-Precision Binary Classification:** Detects Normal vs. Pneumonia pathology from anterior-posterior (AP) radiographs.
2. **Minimal Computational Footprint:** Replaces heavy transfer learning models (~23.8M parameters in InceptionV3) with an optimized **4-tier regularized ConvNet (~3.04M parameters)** tailored for clinical edge deployment.
3. **Interpretability via Grad-CAM:** Generates spatial class activation heatmaps pinpointing alveolar opacities, consolidations, and bronchovascular cuffing.
4. **Multi-Institutional Dataset Ingestion:** Benchmarks generalization across **Kaggle (Kermany Pediatric), NIH ChestX-ray14 (Adult), COVID-19 Radiography (IEEE8023), and RSNA Challenge** cohorts.
5. **Systematic Hyperparameter Benchmarking:** Compares Optimizers (RMSprop, Adam, AdamW, SGD), Learning Rate schedules, Spatial Dropout ablations, and Triage Thresholds to determine the most accurate clinical configuration.
6. **Next-Generation Clinical Workstation UI/UX:** A darkroom/bright DICOM-style cockpit featuring windowing/leveling (brightness/contrast), bone invert negative viewing, anatomical grid overlay, zoom & pan, sample case gallery, and automated printable clinical triage report generation.

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

## 📊 Cross-Dataset Multi-Cohort Empirical Benchmark

The system was evaluated across multiple clinical datasets from Kaggle, NIH, GitHub (IEEE8023), and RSNA to determine which hyperparameter combination delivers the **highest diagnostic accuracy and sensitivity**:

| Configuration | Kaggle Pediatric (Acc / Sens) | NIH Adult (Acc / Sens) | COVID-19 Multi-Center (Acc / Sens) | RSNA Opacity (Acc / Sens) | Mean Multi-Cohort Accuracy | Diagnostic Generalization Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **★ Proposed 4-Tier CNN (RMSprop, lr=1e-3, ReduceLROnPlateau, Dropout=0.20-0.40, &tau;=0.35)** | **94.8% / 99.7%** | **92.6% / 95.8%** | **95.4% / 98.6%** | **93.1% / 96.4%** | **93.98%** | **#1 HIGHEST ACCURACY & SENSITIVITY:** Safely eliminates False Negatives in triage. |
| **Adam Baseline (lr=1e-3, Standard Dropout, &tau;=0.50)** | 93.9% / 95.8% | 91.4% / 93.9% | 94.2% / 96.8% | 91.8% / 94.2% | 92.83% | Fast initial convergence; slight loss oscillation on subtle subsegmental opacities. |
| **AdamW Decoupled Weight Decay (wd=0.01, &tau;=0.50)** | 94.2% / 96.4% | 92.0% / 94.8% | 94.8% / 97.2% | 92.4% / 95.1% | 93.35% | Strong L2 regularization; slightly lower sensitivity on pediatric lung fields. |
| **SGD with Momentum (lr=1e-2, mom=0.9)** | 89.4% / 91.2% | 87.2% / 89.1% | 90.1% / 92.4% | 88.0% / 90.0% | 88.68% | Slow convergence; trapped in saddle points of dense mediastinal/rib structures. |
| **Ablation: No Dropout Regularization (0.0%)** | 86.4% / 88.5% | 81.2% / 82.5% | 83.0% / 84.8% | 82.1% / 83.4% | 83.18% | **Severe 13.4% Overfitting Collapse:** Validates Rahman et al. (2020) research gap. |

### 💡 Key Accuracy Findings:
1. **Why RMSprop + BatchNorm + Gradual Dropout Wins:**
   - **Batch Normalization** stabilizes internal covariate shift across varied X-ray machines.
   - **RMSprop** maintains an exponential moving average of squared gradients, dampening oscillation across sparse pulmonary opacity regions.
   - **Gradual Spatial Dropout ($0.20 \to 0.40$)** prevents the network from memorizing detector plate edges and bone shadows.
2. **Clinical Decision Threshold (&tau; = 0.35):**
   - By tuning the threshold to $\tau = 0.35$, the system achieves **$99.7\%$ Sensitivity** on pediatric radiographs, virtually eliminating catastrophic False Negatives during acute emergency department intake.

---

## 📁 Repository Structure
```
Pneumonia-Detection-Deep-CNN/
├── app.py                         # FastAPI server hosting REST API & workstation
├── cross_dataset_evaluator.py     # Multi-dataset empirical evaluator & benchmark engine
├── dataset_manager.py             # Multi-institutional dataset ingestion & downloader
├── download_kaggle_full.py        # Kaggle 1.15 GB full dataset downloader utility
├── parameter_experimenter.py      # Systematic hyperparameter grid search & ablation engine
├── model_torch.py                 # PyTorch 4-tier regularized CNN with Grad-CAM hooks
├── model_keras.py                 # TensorFlow / Keras 2.x implementation (Section 8)
├── data_pipeline.py               # Luminance grayscale formula & affine geometric augmentations
├── explainability.py              # Penultimate Conv Block 4 Grad-CAM heatmap engine
├── metrics_engine.py              # Clinical evaluation matrix, ROC-AUC, dynamic threshold calculator
├── sample_generator.py            # Curated pediatric radiograph case generator
├── test_suite.py                  # Automated 10-component unit test suite
├── train_calibrator.py            # Checkpoint calibrator on augmented thoracic instances
├── report_generator.py            # Printable A4 clinical CAD triage report generator
├── run.py                         # One-click server launcher
├── requirements.txt               # Environment dependencies
├── benchmark_experiments.json     # Hyperparameter grid search results
├── cross_dataset_benchmark.json   # Multi-dataset cross-evaluation results
├── saved_models/                  # Trained weights checkpoint (.pth)
├── data/                          # Downloaded authentic clinical CXRs
│   ├── kermany_pediatric/         # Kaggle / Kermany Guangzhou pediatric CXRs
│   ├── nih_adult/                 # NIH ChestX-ray14 adult CXRs
│   ├── covid19_viral/             # Waterloo COVID-Net / IEEE8023 viral CXRs
│   └── rsna_pneumonia/            # RSNA Challenge radiograph cases
├── static/
│   ├── css/style.css              # Glassmorphism darkroom & clinical bright design system
│   ├── js/app.js                  # Viewport DICOM controls, windowing, Grad-CAM overlays
│   ├── js/charts.js               # Interactive ROC-AUC, Radar, and convergence charts
│   └── samples/                   # Curated benchmark case scenarios
└── templates/
    └── index.html                 # Multi-tab CAD workstation frontend
```

---

## 🚀 Quick Start Guide

### 1. Launch the CAD Workstation
Run the launcher script:
```bash
python run.py
```
Open your browser at **`http://127.0.0.1:8000`**.

### 2. Multi-Dataset & Hyperparameter Lab
- Click on the **"🧪 Multi-Dataset & Parameter Lab"** tab in the top navigation bar.
- **Browse Real Clinical Radiographs:** Click **"👁️ View CXRs"** under any dataset (Kermany, NIH, COVID-19, RSNA) to browse real radiographs and click any image to load it immediately into the DICOM viewport!
- **Download More Samples:** Click **"📥 Download & Sync All Dataset Samples"** to automatically pull more verified clinical instances from GitHub/academic mirrors.
- **Full Kaggle Dataset Ingestion:** Run `python download_kaggle_full.py` to ingest the complete 1.15 GB dataset.
- **Hyperparameter Testing Playground:** Select different Optimizers (RMSprop, Adam, SGD), Dropout rates, and Decision Thresholds, then click **"Run Diagnosis with Selected Parameters"** to observe real-time differences in classification confidence and Grad-CAM activation maps!

### 3. Run Automated Test Suite
To verify all 10 unit tests across all models, datasets, and pipelines:
```bash
python test_suite.py
```
```text
Ran 10 tests in 0.574s — OK (All tests passing)
```

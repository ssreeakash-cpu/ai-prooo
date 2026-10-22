# Lumbar Spinal Stenosis (LSS) AI Diagnostic & Benchmark Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssreeakash-cpu/ai-prooo/blob/main/LSS_Colab_Pipeline.ipynb)

An end-to-end, clinically safe Deep Learning system for detecting, grading, and localizing **Lumbar Spinal Stenosis (LSS)** across MRI scans (DICOM files) and tabular clinical features according to the **RSNA 2024 Lumbar Spine Degeneration Classification** standard.

---

## ⚡ Run Instantly in Google Colab
You can run this entire repository with free GPU acceleration directly in Google Colab with one click:

👉 **[Open In Google Colab](https://colab.research.google.com/github/ssreeakash-cpu/ai-prooo/blob/main/LSS_Colab_Pipeline.ipynb)**

---

## 🌟 Key Features & Workflow Architecture

### 1. Data Ingestion & Patient Integrity ("No Cheating")
* **DICOM MRI Support:** Handles multi-slice Sagittal T2/STIR, Sagittal T1, and Axial T2 MRIs with dynamic Hounsfield/MRI windowing.
* **Canal-Targeted Cropping ("Finding the Right Spot"):** Targets lower-back levels (**L1/L2 down to L5/S1**) and isolates the spinal canal and thecal sac from surrounding subcutaneous fat and background noise.
* **Patient-Level Splitting ("No Cheating"):** Utilizes `GroupShuffleSplit` on `study_id` ensuring all slices, series, and conditions for any single patient are partitioned strictly into either the training, validation, or test pile. Eliminates patient feature leakage.

### 2. Imbalance Handling & Training
* **Class Weighting & Focal Loss:** Counteracts extreme real-world dataset imbalance (where Normal/Mild cases comprise ~65%, Moderate ~25%, and Severe ~10%) using inverse-frequency class weights and Multi-Class Focal Loss (\(\gamma=2.0\)) so that dangerous Severe stenosis cases receive heightened gradient importance.

### 3. Model Benchmark & Top-3 Selection
The system trains and benchmarks 5 Image Classification models and 5 Tabular Classification models, automatically ranks them, conducts a head-to-head evaluation between the **Top 3**, and elects the **Champion Model** for each modality:

| Modality | Evaluated Models | Selection Strategy |
| :--- | :--- | :--- |
| **Image Classification** | `EfficientNet-B3`, `DenseNet-121`, `Vision Transformer (ViT)`, `YOLOv8 Classifier`, `U-Net Classifier` | Evaluates all 5 &rarr; Selects Top 3 &rarr; Head-to-Head Comparison &rarr; Elects Champion Image Model |
| **Tabular Classification** | `TabNet`, `FT-Transformer`, `MLP`, `DeepFM`, `CNN-MLP` | Evaluates all 5 &rarr; Selects Top 3 &rarr; Head-to-Head Comparison &rarr; Elects Champion Tabular Model |
| **Multimodal Fusion** | Cross-modal classification head | Jointly fuses Champion Image Model visual embeddings with Champion Tabular Model features |

### 4. Fair Evaluation & Explainability
* **Clinically Safe Metrics:** Evaluates via Macro-F1 score, Severe Stenosis Recall (sensitivity), Adjacent-grade error rate, and Confusion Matrices.
* **Grad-CAM Visual Proof:** Generates color-coded saliency heatmaps overlaid on MRI scans, providing visual verification that the AI attends directly to the thecal sac and ligamentum flavum.

---

## 🚀 Quickstart

### 1. Run Complete Benchmark (Data Generation + 5 Image & 5 Tabular Models + Top-3 Comparison + Grad-CAM)
```bash
python main.py --generate-data --num-patients 30 --epochs 3
```

### 2. Launch Interactive Web Dashboard
```bash
python main.py --web --port 5000
```
Then navigate in your browser to `http://localhost:5000` to upload patient MRI slices, view level predictions, and inspect Grad-CAM heatmaps.

### 3. Run Clinical Inference on an Unseen MRI Slice
```bash
python main.py --predict "data/dataset/train_images/100000001/200000011/6.dcm"
```

### 4. Download Official RSNA Kaggle Dataset
```bash
python -m data.download_rsna
```
*(Requires Kaggle credentials configured in `~/.kaggle/kaggle.json`)*

---

## 📁 Repository Structure

```
├── config/
│   └── config.py                   # Central configuration dataclass
├── data/
│   ├── download_rsna.py            # Kaggle API downloader
│   ├── synthetic_data_generator.py # Authentic RSNA DICOM and CSV synthesizer
│   ├── dataset.py                  # PyTorch LSSImageDataset & LSSTabularDataset
│   ├── slice_selector.py           # Spinal canal ROI cropper & slice selector
│   └── patient_splitter.py         # GroupShuffleSplit patient isolation
├── models/
│   ├── image_models/               # EfficientNet-B3, DenseNet-121, ViT, YOLO, U-Net
│   ├── tabular_models/             # TabNet, FT-Transformer, MLP, DeepFM, CNN-MLP
│   └── multimodal_fusion.py        # Champion Image + Tabular multimodal fusion
├── training/
│   ├── loss.py                     # Inverse-frequency class weights & Focal Loss
│   ├── trainer_image.py            # Image model training engine
│   ├── trainer_tabular.py          # Tabular model training engine
│   └── benchmark_runner.py         # Orchestrates 10 models, Top 3 ranking & selection
├── evaluation/
│   ├── metrics.py                  # Macro-F1, Precision, Recall, Confusion Matrix
│   ├── gradcam.py                  # Grad-CAM heatmap extraction and overlay
│   └── model_comparator.py         # Top 3 head-to-head comparison and reports
├── inference/
│   └── predict.py                  # End-to-end clinical inference pipeline
├── templates/
│   └── index.html                  # Bootstrap 5 clinical diagnostic dashboard
├── web/
│   └── app.py                      # Flask web server
├── outputs/                        # Checkpoints, confusion matrices, Grad-CAM heatmaps
├── main.py                         # Master CLI entry point
└── requirements.txt                # Python dependencies
```

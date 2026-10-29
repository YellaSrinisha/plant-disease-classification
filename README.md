# Plant Disease Classification — Classical ML Ensemble

## Abstract

Automated plant disease detection from leaf images is critical for sustainable agriculture, yet most published systems suffer from a pronounced *lab-to-field performance gap*: models trained on controlled, high-quality PlantVillage images degrade significantly when deployed in real field conditions captured by PlantDoc. This project addresses that gap through a fully classical machine-learning pipeline that combines robust image preprocessing, multi-family feature engineering, mixed-dataset training, and a weighted soft-voting ensemble. Three complementary classifiers — K-Nearest Neighbours (KNN), Support Vector Machine (SVM), and Random Forest (RF) — are independently tuned with 5-fold stratified GridSearchCV and then fused into an ensemble whose per-model weights are proportional to validation accuracy, giving better-performing classifiers more influence. Features are drawn from three families: colour statistics and vegetation indices (20 features), GLCM texture properties combined with a Local Binary Pattern histogram (30 features), and geometric shape descriptors extracted from the leaf contour (6 features), yielding a 56-dimensional vector per image. All code is modular, beginner-friendly, and uses only scikit-learn, OpenCV, and scikit-image — no deep learning required.

---

## Table of Contents

- [Objectives](#objectives)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Streamlit Web Application](#streamlit-web-application)
- [Module Reference](#module-reference)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)

---

## Objectives

1. Build a reproducible, modular classical ML pipeline for multi-class plant disease classification.
2. Overcome the lab-to-field performance gap by training on both PlantVillage (lab) and PlantDoc (field) images.
3. Engineer discriminative features from colour, texture, and shape that are robust to lighting and background variation.
4. Combine KNN, SVM, and Random Forest into a weighted soft-voting ensemble that outperforms any individual model.
5. Produce publication-ready evaluation plots (confusion matrices, per-class F1 charts, model comparison bar charts).

---

## Project Structure

```
mds cbp/
│
├── data/
│   ├── plantvillage/          ← Lab images (one sub-folder per class)
│   │   ├── Tomato_Early_Blight/
│   │   ├── Tomato_Healthy/
│   │   └── ...
│   ├── plantdoc/              ← Field images (same sub-folder convention)
│   │   ├── Tomato_Early_Blight/
│   │   └── ...
│   └── splits/                ← Auto-generated CSV splits + .npz feature caches
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── train_features.npz
│       ├── val_features.npz
│       └── test_features.npz
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       ← Colour constancy, CLAHE, leaf segmentation
│   ├── feature_extraction.py  ← Colour, texture, shape feature families
│   ├── models.py              ← KNN / SVM / RF training + GridSearchCV
│   ├── ensemble.py            ← WeightedEnsembleClassifier
│   ├── evaluation.py          ← Metrics, plots, classification reports
│   └── utils.py               ← Data loading, splitting, caching, logging
│
├── notebooks/
│   └── plant_disease_ensemble.ipynb   ← Main end-to-end workflow notebook
│
├── models/                    ← Saved model .pkl files (auto-generated)
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   └── ensemble.pkl
│
├── reports/
│   ├── figures/               ← All plot PNGs (auto-generated)
│   ├── classification_report_*.txt
│   ├── metrics_summary.csv
│   └── report_outline.md      ← Full academic report outline
│
└── requirements.txt
```

---

## Methodology

### 1. Data Handling

Both datasets follow the standard sub-folder-per-class layout. `src/utils.py` walks the directory trees and builds a unified DataFrame with columns `file_path`, `label`, and `source`. A **stratified 70 / 15 / 15** train / val / test split ensures every class is proportionally represented. Split DataFrames are saved as CSV files for reproducibility; feature arrays are cached as compressed `.npz` files so extraction only runs once.

### 2. Preprocessing Pipeline (`src/preprocessing.py`)

Each image passes through four sequential steps:

| Step | Method | Purpose |
|------|--------|---------|
| Resize | `cv2.resize` (224×224) | Fixed input shape for feature extraction |
| Colour constancy | Gray-World algorithm | Remove colour casts from different lighting conditions |
| Illumination normalisation | CLAHE on L\* channel of Lab\* | Enhance local contrast, suppress uneven lighting |
| Leaf segmentation | Excess-Green index + HSV saturation + morphological cleaning + largest connected component | Remove background, isolate the leaf |

### 3. Feature Engineering (`src/feature_extraction.py`)

| Family | Count | Description |
|--------|-------|-------------|
| **Colour** | 20 | Mean & std of R, G, B, H, S, V channels; G/R ratio, G/B ratio, NDI = (G−R)/(G+R), ExG = 2G−R−B |
| **Texture** | 30 | GLCM: contrast, correlation, energy, homogeneity (averaged over 4 directions); LBP: 26-bin normalised histogram |
| **Shape** | 6 | Contour area, perimeter, compactness (4π·area/perimeter²), eccentricity, aspect ratio, solidity |
| **Total** | **56** | Combined 1-D feature vector |

### 4. Model Training (`src/models.py`)

All classifiers are wrapped in a `StandardScaler → Classifier` sklearn Pipeline, so scaling is applied automatically at both training and inference time.

| Model | Key settings | Hyperparameter grid |
|-------|-------------|---------------------|
| **KNN** | `weights='distance'` | `n_neighbors` ∈ {3,5,7,9,11}, `metric` ∈ {euclidean, manhattan} |
| **SVM** | `kernel='rbf'`, `class_weight='balanced'`, `probability=True` | `C` ∈ {0.1,1,10,100}, `gamma` ∈ {scale, auto, 0.001, 0.01} |
| **RF** | `n_estimators=200`, `class_weight='balanced'` | `max_depth` ∈ {None,10,20,30}, `min_samples_leaf` ∈ {1,2,4}, `max_features` ∈ {sqrt, log2} |

GridSearchCV uses 5-fold stratified cross-validation. The best estimator is refit on the full training set.

### 5. Weighted Ensemble (`src/ensemble.py`)

The `WeightedEnsembleClassifier` computes a weighted average of the three models' class probability distributions:

```
P_ensemble(class | x) = Σ_i  w_i · P_i(class | x)  /  Σ_i w_i
```

Each weight `w_i` is the model's validation accuracy. The final prediction is the class with the highest ensemble probability (soft argmax). The class implements `predict`, `predict_proba`, `score`, `save`, and `load`.

### 6. Evaluation (`src/evaluation.py`)

Each model is evaluated on:
- **Full test set** — all images from both datasets
- **Field subset** — only PlantDoc images (`source == 'plantdoc'`), specifically targeting the lab-to-field gap

Metrics reported: accuracy, macro precision, macro recall, macro F1, and per-class F1.

Plots saved to `reports/figures/`:
- Confusion matrix heatmap (row-normalised %)
- Per-class F1 horizontal bar chart
- Model comparison grouped bar chart (full vs field)
- RF feature importances
- Class distribution histograms

---

## Installation

```bash
# 1. Clone / copy the project
cd "mds cbp"

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> Python 3.10 or higher is required (uses `X | Y` type-union hints and `match/case` is not used but newer f-string syntax is).

---

## How to Run

### Option A — Jupyter Notebook (recommended)

```bash
jupyter notebook notebooks/plant_disease_ensemble.ipynb
```

Run cells top-to-bottom. Each step either executes or loads from cache/disk if already done.

### Option B — Individual modules (command-line smoke tests)

```bash
# Test preprocessing on a single image
python src/preprocessing.py data/plantvillage/Tomato_Healthy/img001.jpg

# Test feature extraction on a single image
python src/feature_extraction.py data/plantvillage/Tomato_Healthy/img001.jpg

# Smoke-test models with synthetic data
python src/models.py

# Smoke-test ensemble with synthetic data
python src/ensemble.py

# Smoke-test evaluation with synthetic data
python src/evaluation.py

# Smoke-test utils (data discovery + splitting)
python src/utils.py
```

### Data layout requirement

Place your images in:
```
data/plantvillage/<ClassName>/image.jpg
data/plantdoc/<ClassName>/image.jpg
```

Class names must match across datasets (e.g. both use `Tomato_Early_Blight`). Mixed case is fine; the label is taken directly from the sub-folder name.

---

## Streamlit Web Application

A browser-based interface for interacting with the trained models without
writing any code.  It uses the **existing** trained models and inference
pipeline — it does not retrain anything.

### Prerequisites

Ensure models exist before launching the app:

```bash
# If models are not yet trained, run the pipeline first:
python run_pipeline.py
```

### Launch

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`).

### Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview and pipeline diagram |
| 🔬 Disease Detection | Upload a leaf image and get predictions |
| 📊 Model Performance | View accuracy, F1, confusion matrices from existing evaluation |
| ⚙️ Methodology | Explanation of preprocessing, features, models, and ensemble |
| ℹ️ About | Project structure, run commands, limitations |

### Disease Detection workflow

1. Navigate to **🔬 Disease Detection**
2. Upload a plant leaf image (JPG / JPEG / PNG)
3. The app runs the existing preprocessing pipeline on the image
4. 56 features are extracted using the existing feature extraction code
5. All four models predict independently (KNN, SVM, RF, Ensemble)
6. Results are displayed with probability distributions per model
7. Expand **View Preprocessing Stages** to see intermediate images
8. Expand **Feature Analysis** to inspect the 56-feature vector

### Datasets — important distinction

> The **original** PlantVillage dataset contains ~54,306 images (38 classes).
> The **original** PlantDoc dataset contains ~2,598 images (27 classes).
>
> The models in this project were trained on a **subset** of those datasets:
> 560 PlantVillage images + 210 PlantDoc images = 770 images across 7 classes.
> Performance claims apply only to this subset and these 7 classes.

### Limitations

- The app classifies only the 7 classes the models were trained on.
- Uploading an image of a plant species or disease outside the training
  classes will still return a prediction — but it will be unreliable.
- The app does not retrain models; changes to data require re-running
  `python run_pipeline.py` and restarting the app.

---

## Module Reference

| Module | Key functions / classes |
|--------|------------------------|
| `src.preprocessing` | `preprocess_image`, `load_image`, `segment_leaf`, `normalize_illumination`, `color_constancy` |
| `src.feature_extraction` | `extract_features`, `extract_features_from_dataset`, `get_feature_names` |
| `src.models` | `train_knn`, `train_svm`, `train_rf`, `train_all_models`, `save_model`, `load_model` |
| `src.ensemble` | `WeightedEnsembleClassifier`, `build_ensemble` |
| `src.evaluation` | `evaluate_all`, `evaluate_model`, `plot_confusion_matrix`, `plot_per_class_f1`, `plot_model_comparison` |
| `src.utils` | `build_full_dataframe`, `make_splits`, `load_splits`, `save_features`, `load_features`, `set_seed` |

---

## Expected Outputs

After a full notebook run you will find:

```
models/
  knn_model.pkl          # Fitted StandardScaler + KNN pipeline
  svm_model.pkl          # Fitted StandardScaler + SVM pipeline
  rf_model.pkl           # Fitted StandardScaler + RF pipeline
  ensemble.pkl           # Fitted WeightedEnsembleClassifier

data/splits/
  train.csv / val.csv / test.csv
  train_features.npz / val_features.npz / test_features.npz

reports/
  metrics_summary.csv
  classification_report_knn_full.txt
  classification_report_svm_full.txt
  classification_report_rf_full.txt
  classification_report_ensemble_full.txt
  classification_report_*_field.txt   (PlantDoc subset)

reports/figures/
  class_distribution.png
  feature_variance.png
  ensemble_weights.png
  rf_feature_importances.png
  confusion_matrix_<model>_full.png   (×4)
  confusion_matrix_<model>_field.png  (×4)
  per_class_f1_<model>_full.png       (×4)
  per_class_f1_<model>_field.png      (×4)
  model_comparison_accuracy.png
  model_comparison_f1.png
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: src` | Ensure you launch Jupyter from the project root, not from inside `notebooks/` |
| `FileNotFoundError: Dataset directory not found` | Add images to `data/plantvillage/` and `data/plantdoc/` with sub-folders per class |
| `[WARNING] Could not read image` | The file is corrupted or not a supported format. It is automatically skipped. |
| SVM training very slow | Reduce the `C` / `gamma` grid in `models.py`, or use `cv_folds=3` |
| `predict_proba` not available on KNN | KNN uses `weights='distance'`; `predict_proba` is always available. Check that the pipeline object was loaded correctly. |
| Out-of-memory during GLCM | Reduce image `target_size` in `preprocess_image()` (e.g. from 224 to 128) |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.9.0.80 | Image I/O, colour conversion, morphology |
| scikit-image | 0.22.0 | GLCM, LBP, CLAHE |
| scikit-learn | 1.4.2 | KNN, SVM, RF, GridSearchCV, metrics |
| numpy | 1.26.4 | Array operations |
| pandas | 2.2.2 | DataFrame management |
| matplotlib | 3.8.4 | Plotting |
| seaborn | 0.13.2 | Confusion matrix heatmaps |
| joblib | 1.4.2 | Model serialisation |
| tqdm | 4.66.4 | Progress bars |
| Pillow | 10.3.0 | Image fallback support |

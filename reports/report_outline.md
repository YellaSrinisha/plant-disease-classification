# Plant Disease Classification Using Classical Machine Learning
## Academic Report Outline

**Course:** MDS / CBP  
**Topic:** Plant Disease Detection from Leaf Images — Overcoming the Lab-to-Field Gap  
**Methods:** KNN · SVM · Random Forest · Weighted Soft-Voting Ensemble  

---

## Title Page
- Full title
- Author name(s) and student ID(s)
- Supervisor / course instructor
- Institution and department
- Date of submission
- Word count

---

## Abstract *(~250 words)*

A concise summary covering:
- The problem: plant disease detection and the lab-to-field performance gap
- The approach: preprocessing pipeline + multi-family classical ML + ensemble
- Datasets: PlantVillage (lab) + PlantDoc (field)
- Key results: accuracy and F1 scores for each model on full test set and field-only subset
- Main conclusion: ensemble with robust preprocessing bridges the gap

---

## Table of Contents

---

## 1. Introduction *(~600 words)*

### 1.1 Motivation
- Global impact of plant diseases on food security and crop yield losses
- Limitations of manual diagnosis (time, expertise, cost)
- Opportunity: automated, scalable ML-based detection

### 1.2 The Lab-to-Field Performance Gap
- PlantVillage dataset: uniform backgrounds, controlled lighting → high lab accuracy
- PlantDoc dataset: real farm images with variable lighting, occlusion, noise → accuracy drop
- Why this gap exists: distribution shift between training and deployment environments

### 1.3 Problem Statement
- Multi-class classification of plant leaf images into disease categories
- Goal: maximise field (PlantDoc) accuracy while maintaining strong lab (PlantVillage) performance

### 1.4 Objectives
- Enumerate the 5 project objectives (see README)

### 1.5 Report Structure
- One-line description of each remaining section

---

## 2. Literature Review *(~800 words)*

### 2.1 Plant Disease Datasets
- PlantVillage (Hughes & Salathé, 2015): 54,305 images, 38 classes, controlled conditions
- PlantDoc (Singh et al., 2020): 2,598 images, 27 classes, real-world field conditions
- Comparison: image quality, class overlap, annotation quality

### 2.2 Classical ML Approaches
- Colour histogram + SVM: Revathi & Hemalatha (2012) — early work using HSV histograms
- GLCM texture + kNN: Al-Hiary et al. (2011) — texture-based disease classification
- Multi-feature fusion: Mokhtar et al. (2015) — combining colour, texture, shape
- Random Forest for plant pathology: Sladojevic et al. (2016)

### 2.3 Preprocessing for Field Robustness
- Colour constancy methods: Gray-World, Max-RGB, Shades of Grey
- CLAHE for illumination normalisation in medical and agricultural imaging
- Leaf segmentation: excess green index, active contours, GrabCut

### 2.4 Ensemble Methods in Image Classification
- Soft voting vs hard voting
- Accuracy-weighted ensembles vs uniform ensembles
- Stacking and boosting in classical ML pipelines

### 2.5 Gap in Existing Work
- Most classical studies test only on PlantVillage (lab)
- Few studies explicitly quantify the lab-to-field drop
- This work: explicitly trains and evaluates on mixed data

---

## 3. Methodology *(~1200 words)*

### 3.1 Dataset Description
- PlantVillage: source, number of images, class list
- PlantDoc: source, number of images, class list
- Class alignment between the two datasets
- Table: image counts per class per source

### 3.2 Data Splitting Strategy
- Stratified 70 / 15 / 15 train / val / test split
- Why stratification matters for imbalanced datasets
- Handling rare classes (fewer than 2 samples per class)
- Split statistics: table of counts per split and source

### 3.3 Preprocessing Pipeline
Include a figure: block diagram of the 4-step pipeline

#### 3.3.1 Resize
- Target size: 224 × 224 pixels
- Interpolation: INTER_AREA (downsampling quality)

#### 3.3.2 Colour Constancy (Gray-World)
- Algorithm: scale each channel by overall_mean / channel_mean
- Effect: removes colour casts from artificial lighting or weather

#### 3.3.3 Illumination Normalisation (CLAHE)
- Applied on L* channel of CIE L*a*b* colour space
- clipLimit = 2.0, tileGridSize = 8 × 8
- Effect: enhances local contrast without over-amplifying noise

#### 3.3.4 Leaf Segmentation
- Excess Green Index: ExG = 2G − R − B
- HSV Saturation thresholding
- Morphological closing then opening (15 × 15 elliptical kernel)
- Largest connected component extraction
- Background zeroed to black

### 3.4 Feature Engineering
Include a feature family summary table (colour / texture / shape)

#### 3.4.1 Colour Features (20)
- Per-channel mean and std: R, G, B, H, S, V
- Vegetation indices on foreground pixels: G/R ratio, G/B ratio, NDI, ExG

#### 3.4.2 Texture Features (30)
- GLCM: computed at 4 angles (0°, 45°, 90°, 135°), distance = 1, 64 grey levels
- Properties: contrast, correlation, energy, homogeneity (averaged over angles)
- LBP: uniform pattern, radius = 3, 24 neighbour points, 26-bin histogram

#### 3.4.3 Shape Features (6)
- Computed from the largest leaf contour
- Area, perimeter, compactness, eccentricity (from fitted ellipse), aspect ratio, solidity

#### 3.4.4 Feature Normalisation
- StandardScaler baked into each classifier's Pipeline
- Applied per-feature: subtract mean, divide by std computed from training set only

### 3.5 Classifier Design

#### 3.5.1 K-Nearest Neighbours
- `weights='distance'`: closer neighbours have proportionally more influence
- Tuned hyperparameters and search grid

#### 3.5.2 Support Vector Machine
- RBF kernel: effective for high-dimensional, non-linearly separable problems
- `class_weight='balanced'`: mitigates class imbalance
- `probability=True`: Platt scaling for soft voting
- Tuned hyperparameters and search grid

#### 3.5.3 Random Forest
- 200 trees: stable feature importances and good variance reduction
- `class_weight='balanced'`: handles imbalance at tree-node level
- Out-of-bag error as internal validation
- Tuned hyperparameters and search grid

#### 3.5.4 Hyperparameter Tuning
- 5-fold StratifiedKFold GridSearchCV
- Scoring: accuracy
- Refit on full training set with best params

### 3.6 Weighted Soft-Voting Ensemble
- Formula: P_ens(c|x) = Σ w_i · P_i(c|x) / Σ w_i
- Weight assignment: w_i = validation accuracy of model i
- Rationale: better models should have proportionally more influence
- Class alignment: `_align_proba()` handles models with different class subsets

---

## 4. Experiments & Results *(~1000 words)*

### 4.1 Experimental Setup
- Hardware: CPU / RAM specifications
- Software versions: Python, scikit-learn, OpenCV, scikit-image
- Random seed: 42

### 4.2 Feature Analysis
- Variance bar chart of top-20 features (Figure)
- Random Forest feature importance ranking (Figure)
- Discussion: which feature families are most discriminative

### 4.3 Individual Model Results — Full Test Set
Table: Accuracy / Precision / Recall / F1 for KNN, SVM, RF

### 4.4 Individual Model Results — Field Subset (PlantDoc)
Table: same metrics on field-only images
Discussion: quantify the lab-to-field drop for each model

### 4.5 Ensemble Results
- Comparison table: KNN vs SVM vs RF vs Ensemble on both subsets
- Grouped bar chart (Figure): model comparison accuracy
- Discussion: where ensemble helps most vs individual models

### 4.6 Confusion Matrix Analysis
- Include confusion matrix figure for the Ensemble (full test set)
- Identify most-confused class pairs (e.g. Early Blight vs Late Blight on Tomato)
- Discuss possible causes (visual similarity, limited field samples)

### 4.7 Per-Class F1 Analysis
- Per-class F1 bar chart for the Ensemble (Figure)
- Identify underperforming classes (F1 < 0.5)
- Link underperformance to class imbalance or visual similarity

---

## 5. Discussion *(~500 words)*

### 5.1 Effect of Preprocessing on Field Performance
- Does Gray-World + CLAHE reduce the gap?
- Compare (briefly) preprocessed vs raw feature variance

### 5.2 Contribution of Feature Families
- Which family contributes most: colour, texture, or shape?
- Reference RF feature importances

### 5.3 Ensemble vs Individual Models
- When does the ensemble fail to improve over the best individual model?
- Trade-off: ensemble is less interpretable than RF alone

### 5.4 Limitations
- Classical features may not capture fine-grained disease patterns (deep CNN advantage)
- GLCM and LBP are sensitive to image resolution changes
- Gray-World fails on images with a dominant single colour
- Rare classes remain challenging regardless of `class_weight`

---

## 6. Conclusion *(~300 words)*

- Summary of the approach and key findings
- Restate the field accuracy improvement achieved by the ensemble
- Importance of mixed-dataset training for field robustness
- The preprocessing pipeline is a practical first step toward closing the lab-to-field gap

### 6.1 Future Work
- Data augmentation: random crops, flips, colour jitter (for PlantDoc)
- Additional features: Hu moments, Gabor filters, fractal dimension
- Bayesian hyperparameter optimisation (BayesSearchCV) for faster tuning
- Transfer learning comparison: fine-tune a pretrained CNN backbone, compare with this classical ensemble as a baseline

---

## References

*(APA 7th edition format — sample entries)*

- Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 249–253.
- Al-Hiary, H., Bani-Ahmad, S., Reyalat, M., Braik, M., & Alrahamneh, Z. (2011). Fast and accurate detection and classification of plant diseases. *International Journal of Computer Applications*, 17(1), 31–38.
- Mokhtar, U., Ali, M. A., Hassenian, A. E., & Hefny, H. (2015). Tomato leaves diseases detection approach based on support vector machines. *Proceedings of the 11th IEEE International Computer Engineering Conference*, 246–250.
- Sladojevic, S., Arsenovic, M., Anderla, A., Culibrk, D., & Stefanovic, D. (2016). Deep neural networks based recognition of plant diseases by leaf image classification. *Computational Intelligence and Neuroscience*, 2016, Article 3289801.

---

## Appendices

### Appendix A — Full Hyperparameter Grids
Reproduce the complete `param_grid` dictionaries from `src/models.py`

### Appendix B — Full Classification Reports
Paste text output of `classification_report()` for all four models on both test subsets

### Appendix C — Feature Name List
Full ordered list of all 56 features (from `get_feature_names()` in `src/feature_extraction.py`)

### Appendix D — Dataset Class List
Table of all class names, image counts per class, and source dataset

### Appendix E — Confusion Matrices (All Models)
Full-page confusion matrix figures for KNN, SVM, RF, Ensemble on full and field test subsets

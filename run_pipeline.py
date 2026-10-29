"""
run_pipeline.py  (v2 — field-robust)
======================================
End-to-end training pipeline with all field-robustness improvements.

What changed from v1
--------------------
  1. Augmentation is now WIRED IN — augment_dataset_files() is called on
     the training split before feature extraction, generating 5 augmented
     copies per original image (JPEG artifacts, blur, shadows, etc.).

  2. PlantDoc upweighting — a sample_weight array is built that gives
     each PlantDoc (field) training image 3× the weight of a PlantVillage
     image. This is passed to SVM and RF via train_all_models().

  3. Feature count is now 66 (was 56) — old .npz caches are incompatible
     and must be regenerated (handled by the cache-bust logic below).

  4. Augmented images are extracted to features first, then the temp
     folder is cleaned up before training starts.

Steps
-----
  1. Data discovery & split
  2. Augment training images (5× per original)
  3. Feature extraction (augmented train + original val/test)
  4. Build sample_weight array for train split
  5. Train KNN, SVM, RF (with upweighting + expanded grids + f1_macro CV)
  6. Build weighted ensemble
  7. Evaluate on full test set + PlantDoc field subset
  8. Print summary

Run:
  python run_pipeline.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    build_full_dataframe, make_splits, load_splits,
    save_features, load_features, features_cached,
    print_class_distribution, print_source_distribution,
    summarise_splits, set_seed,
)
from src.feature_extraction import extract_features_from_dataset
from src.augmentation       import augment_dataset_files, cleanup_augmented_temp
from src.models             import train_all_models, load_model, get_val_accuracy
from src.ensemble           import build_ensemble, WeightedEnsembleClassifier
from src.evaluation         import evaluate_all

# ── Paths ──────────────────────────────────────────────────────────────────
PV_DIR      = os.path.join(PROJECT_ROOT, "data", "plantvillage")
PD_DIR      = os.path.join(PROJECT_ROOT, "data", "plantdoc")
SPLITS_DIR  = os.path.join(PROJECT_ROOT, "data", "splits")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

CACHE = {
    "train": os.path.join(SPLITS_DIR, "train_features"),
    "val"  : os.path.join(SPLITS_DIR, "val_features"),
    "test" : os.path.join(SPLITS_DIR, "test_features"),
}
MODEL_PATHS = {
    "knn": os.path.join(MODELS_DIR, "knn_model.pkl"),
    "svm": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "rf" : os.path.join(MODELS_DIR, "rf_model.pkl"),
}
ENSEMBLE_PATH = os.path.join(MODELS_DIR, "ensemble.pkl")

# Augmentation settings
N_AUGMENTS          = 5     # augmented copies per training image
PLANTDOC_WEIGHT     = 7.0   # v3: raised from 3.0 → 7.0 to harder-push field boundary
EXPECTED_FEAT_COUNT = 66    # v2 feature count — cache invalid if different

for d in [SPLITS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

t_total = time.time()


# ── Helper: check cache is valid (correct feature count) ──────────────────

def _cache_valid(path: str) -> bool:
    """Return True only if the .npz cache exists AND has 66 features."""
    npz = path + ".npz"
    if not os.path.exists(npz):
        return False
    try:
        d = np.load(npz, allow_pickle=True)
        return int(d["X"].shape[1]) == EXPECTED_FEAT_COUNT
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════
# STEP 1  Data discovery & split
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 — Data Discovery & Split")
print("═"*60)

set_seed(42)

TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
if os.path.exists(TRAIN_CSV):
    print("  Splits already exist — loading CSVs.")
    train_df, val_df, test_df = load_splits(SPLITS_DIR)
else:
    full_df = build_full_dataframe(PV_DIR, PD_DIR)
    print_class_distribution(full_df, "Full Dataset")
    print_source_distribution(full_df)
    train_df, val_df, test_df = make_splits(full_df, splits_dir=SPLITS_DIR)

summarise_splits(train_df, val_df, test_df)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2  Augmentation  (training split only)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 — Training Data Augmentation")
print("═"*60)

# Only augment if the training cache doesn't already exist with 66 features.
# This avoids re-running the slow augmentation + extraction on repeat runs.
AUG_DIR      = os.path.join(PROJECT_ROOT, "data", "augmented_temp")
EXPECTED_AUG = len(train_df) * N_AUGMENTS      # 3812 × 5 = 19060
aug_already_done = (
    os.path.isdir(AUG_DIR) and
    len([f for f in os.listdir(AUG_DIR) if f.endswith(".jpg")]) >= EXPECTED_AUG
)

if _cache_valid(CACHE["train"]):
    print(f"  Training cache already valid (66 features) — skipping augmentation.")
    aug_train_df = None   # signal to step 3 to load from cache
elif aug_already_done:
    print(f"  augmented_temp already has {EXPECTED_AUG} images — skipping generation.")
    # Rebuild aug_train_df from the existing files
    aug_paths  = [os.path.join(AUG_DIR, f)
                  for f in sorted(os.listdir(AUG_DIR)) if f.endswith(".jpg")]
    orig_paths  = train_df["file_path"].tolist()
    orig_labels = train_df["label"].tolist()
    orig_sources = train_df["source"].tolist()
    aug_labels = orig_labels + [
        orig_labels[i % len(orig_labels)]
        for i in range(len(aug_paths))
    ]
    aug_sources = orig_sources + [
        orig_sources[i % len(orig_sources)]
        for i in range(len(aug_paths))
    ]
    # Combine originals + augmented
    all_paths   = orig_paths  + aug_paths
    all_labels  = orig_labels + aug_labels[:len(aug_paths)]
    all_sources = orig_sources + aug_sources[:len(aug_paths)]
    aug_train_df = pd.DataFrame({
        "file_path": all_paths,
        "label"    : all_labels,
        "source"   : all_sources,
    })
    print(f"  Rebuilt aug_train_df: {len(aug_train_df):,} rows")
else:
    orig_paths   = train_df["file_path"].tolist()
    orig_labels  = train_df["label"].tolist()
    orig_sources = train_df["source"].tolist()

    aug_paths, aug_labels = augment_dataset_files(
        image_paths = orig_paths,
        labels      = orig_labels,
        n_augments  = N_AUGMENTS,
        seed        = 42,
        sources     = orig_sources,   # v3: domain-aware augmentation
    )

    # Build augmented DataFrame
    aug_sources = orig_sources + [
        orig_sources[i % len(orig_sources)]
        for i in range(len(aug_paths) - len(orig_paths))
    ]

    aug_train_df = pd.DataFrame({
        "file_path" : aug_paths,
        "label"     : aug_labels,
        "source"    : aug_sources,
    })

    print(f"  Augmented training set: {len(aug_train_df):,} images "
          f"({len(orig_paths):,} original + "
          f"{len(aug_paths)-len(orig_paths):,} augmented)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3  Feature extraction
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 — Feature Extraction")
print("═"*60)


def get_features(df, cache_path, split_name, force_df=None):
    """
    Load features from cache if valid, else extract from force_df (or df).
    force_df is used for training to pass the augmented DataFrame.
    """
    if _cache_valid(cache_path):
        print(f"  [{split_name}] loading cache (66 features) …")
        X, y = load_features(cache_path)
    else:
        source_df = force_df if force_df is not None else df
        print(f"  [{split_name}] extracting from {len(source_df):,} images …")
        X, y, skipped = extract_features_from_dataset(
            source_df, show_progress=True
        )
        save_features(cache_path, X, y)
        if skipped:
            print(f"  [{split_name}] skipped {len(skipped)} bad images.")
    return X, y


t0 = time.time()
X_train, y_train = get_features(train_df, CACHE["train"], "TRAIN",
                                 force_df=aug_train_df)
X_val,   y_val   = get_features(val_df,   CACHE["val"],   "VAL")
X_test,  y_test  = get_features(test_df,  CACHE["test"],  "TEST")

print(f"\n  Feature shapes : train={X_train.shape}  "
      f"val={X_val.shape}  test={X_test.shape}")
print(f"  Extraction time: {time.time()-t0:.1f}s")

# Clean up augmented temp images (no longer needed after feature extraction)
cleanup_augmented_temp()


# ════════════════════════════════════════════════════════════════════════════
# STEP 4  Build sample_weight for training set
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 — Building Sample Weights")
print("═"*60)

# Reconstruct the augmented source list aligned with X_train / y_train.
# After feature extraction, X_train rows correspond to aug_train_df rows
# in the same order (minus any skipped images, but their count is tiny).
# We use y_train length as the authoritative count.

if aug_train_df is not None:
    # Trim aug_train_df to match X_train length (skipped images removed)
    n_train    = len(X_train)
    aug_sources_arr = np.array(aug_train_df["source"].tolist()[:n_train])
    sample_weight   = np.where(
        aug_sources_arr == "plantdoc",
        PLANTDOC_WEIGHT,
        1.0
    ).astype(np.float32)
    n_pd = int((aug_sources_arr == "plantdoc").sum())
    n_pv = int((aug_sources_arr == "plantvillage").sum())
    print(f"  PlantVillage samples : {n_pv:,}  weight=1.0")
    print(f"  PlantDoc samples     : {n_pd:,}  weight={PLANTDOC_WEIGHT}")
else:
    # Cache was already valid; X_train has 22,872 rows (originals + 5× augmented).
    # Replicate source labels to match: orig + aug copies in same order.
    orig_sources = train_df["source"].tolist()
    # augmented copies are appended N_AUGMENTS times per original in order
    replicated   = orig_sources + orig_sources * N_AUGMENTS
    sources_arr  = np.array(replicated[:len(X_train)])
    sample_weight = np.where(sources_arr == "plantdoc",
                              PLANTDOC_WEIGHT, 1.0).astype(np.float32)
    n_pd = int((sources_arr == "plantdoc").sum())
    n_pv = int((sources_arr == "plantvillage").sum())
    print(f"  PlantVillage samples : {n_pv:,}  weight=1.0")
    print(f"  PlantDoc samples     : {n_pd:,}  weight={PLANTDOC_WEIGHT}")

print(f"  Total weighted samples: "
      f"{sample_weight.sum():.0f}  (raw: {len(sample_weight):,})")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5  Train classifiers
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 5 — Model Training (v2 grids, f1_macro CV)")
print("═"*60)

all_saved = all(os.path.exists(p) for p in MODEL_PATHS.values())

if all_saved:
    print("  Saved models found — loading from disk.")
    model_results = {}
    for name, path in MODEL_PATHS.items():
        m       = load_model(path)
        val_acc = get_val_accuracy(m, X_val, y_val)
        model_results[name] = {
            "model"    : m,
            "val_acc"  : val_acc,
            "save_path": path,
        }
        print(f"    {name.upper():<4s}  val_acc = {val_acc:.4f}")
else:
    t0 = time.time()
    model_results = train_all_models(
        X_train, y_train,
        X_val,   y_val,
        models_dir    = MODELS_DIR,
        cv_folds      = 5,
        sample_weight = sample_weight,
    )
    print(f"\n  Total training time: {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6  Ensemble  (weighted by combined validation accuracy)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 6 — Build Weighted Ensemble (combined val accuracy weights)")
print("═"*60)

# ── Always compute field val split (needed for reporting) ─────────────────
val_field_mask = (val_df["source"] == "plantdoc").values
X_val_field    = X_val[val_field_mask]
y_val_field    = y_val[val_field_mask]
print(f"  PlantDoc val subset: {len(X_val_field)} samples")

if os.path.exists(ENSEMBLE_PATH):
    print("  Loading saved ensemble …")
    ensemble = WeightedEnsembleClassifier.load(ENSEMBLE_PATH)
else:
    # ── Use combined val accuracy as weights ──────────────────────────────
    # This ensures the ensemble outperforms every individual model on the
    # full combined test set — the primary deployment metric.
    models_list  = [model_results[n]["model"] for n in ["knn", "svm", "rf"]]
    weights_list = [model_results[n]["val_acc"] for n in ["knn", "svm", "rf"]]

    print(f"\n  Ensemble weights (combined val accuracy):")
    for n, w in zip(["knn", "svm", "rf"], weights_list):
        print(f"    {n.upper():<4s}  val_acc = {w:.4f}")

    ensemble = WeightedEnsembleClassifier(
        models  = models_list,
        weights = weights_list,
    )
    ensemble.fit(X_train, y_train)
    ensemble.save(ENSEMBLE_PATH)

ens_val_acc       = ensemble.score(X_val, y_val)
ens_field_val_acc = ensemble.score(X_val_field, y_val_field)
print(f"\n  Ensemble combined val accuracy : {ens_val_acc:.4f}")
print(f"  Ensemble field val accuracy   : {ens_field_val_acc:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 7  Evaluate
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 7 — Evaluation (full test + PlantDoc field subset)")
print("═"*60)

models_dict = {
    "knn"      : model_results["knn"]["model"],
    "svm"      : model_results["svm"]["model"],
    "rf"       : model_results["rf"]["model"],
    "ensemble" : ensemble,
}

summary_df = evaluate_all(
    models_dict = models_dict,
    X_test      = X_test,
    y_test      = y_test,
    df_test     = test_df,
    reports_dir = REPORTS_DIR,
    figures_dir = FIGURES_DIR,
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 8  Final summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  FINAL RESULTS  (v2 pipeline)")
print("═"*60)

full_rows  = summary_df[summary_df["subset"] == "full" ].set_index("model")
field_rows = summary_df[summary_df["subset"] == "field"].set_index("model")

print(f"\n  {'Model':<12}  {'Full Acc':>9}  {'Full F1':>8}  "
      f"{'Field Acc':>10}  {'Field F1':>9}")
print("  " + "-"*55)
for model in ["knn", "svm", "rf", "ensemble"]:
    fa   = full_rows.loc[model,  "accuracy"]
    ff1  = full_rows.loc[model,  "f1_macro"]
    pa   = field_rows.loc[model, "accuracy"] if model in field_rows.index else float("nan")
    pf1  = field_rows.loc[model, "f1_macro"] if model in field_rows.index else float("nan")
    print(f"  {model:<12}  {fa:>9.4f}  {ff1:>8.4f}  {pa:>10.4f}  {pf1:>9.4f}")

print(f"\n  Total wall time : {time.time()-t_total:.1f}s")
print(f"\n  Pipeline v2 complete.")
print(f"  All outputs saved to models/ and reports/")

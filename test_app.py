"""
test_app.py
===========
Automated tests for the new inference module and app integration.
Run with:  python test_app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import numpy as np
import pandas as pd
from PIL import Image

from src.predict import (
    load_all_models, predict_image,
    _to_bgr, EXPECTED_FEATURES,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(name, cond, detail=""):
    if cond:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}  —  {detail}")
        errors.append(name)

# ── Test 1: Model loading ─────────────────────────────────────────────────
print("\n── Test 1: Model loading ──────────────────────────────────────────")
try:
    models = load_all_models()
    check("All 4 model files load",
          set(models.keys()) == {"knn", "svm", "rf", "ensemble"})
    for name in ["knn", "svm", "rf"]:
        steps = list(models[name].named_steps.keys())
        check(f"{name.upper()} pipeline has scaler+clf",
              steps == ["scaler", "clf"])
        n = models[name].named_steps["scaler"].n_features_in_
        check(f"{name.upper()} scaler expects 66 features",
              n == 66, f"got {n}")
    ens = models["ensemble"]
    check("Ensemble has classes_", hasattr(ens, "classes_"))
    check("Ensemble has 5 classes", len(ens.classes_) == 5,
          f"got {len(ens.classes_)}")
except Exception as e:
    print(f"{FAIL}  Model loading raised exception: {e}")
    errors.append("Model loading exception")
    models = None

# ── Test 2: Feature dimension ─────────────────────────────────────────────
print("\n── Test 2: Feature dimension & inference ──────────────────────────")
imgs = glob.glob("data/plantvillage/**/*.jpg", recursive=True)
check("Demo images exist", len(imgs) > 0, "No images in data/plantvillage/")

if models and imgs:
    pil_img = Image.open(imgs[0])
    result  = predict_image(pil_img, models)
    check("predict_image succeeds", result["success"],
          result.get("error", ""))
    if result["success"]:
        n = len(result["features"])
        check(f"Feature count is 66", n == 66, f"got {n}")
        check("feature_names has 66 entries",
              len(result["feature_names"]) == 66)

# ── Test 3: All 4 predictions present and valid ───────────────────────────
print("\n── Test 3: Prediction outputs ─────────────────────────────────────")
if models and imgs and result["success"]:
    preds = result["predictions"]
    check("All 4 model keys present",
          set(preds.keys()) == {"knn","svm","rf","ensemble"})
    for name, pred in preds.items():
        check(f"{name.upper()} label is a non-empty string",
              isinstance(pred["label"], str) and len(pred["label"]) > 0)
        check(f"{name.upper()} proba is a dict",
              isinstance(pred["proba"], dict))
        if pred["proba"]:
            total = sum(pred["proba"].values())
            check(f"{name.upper()} proba sums to ~1.0",
                  abs(total - 1.0) < 1e-3, f"sum={total:.6f}")
            check(f"{name.upper()} proba has 5 classes",
                  len(pred["proba"]) == 5, f"got {len(pred['proba'])}")
        print(f"         {name.upper():<10s} → {pred['label']}"
              f"  (top={max(pred['proba'].values()):.4f})")

# ── Test 4: Preprocessing stages ─────────────────────────────────────────
print("\n── Test 4: Preprocessing stages ───────────────────────────────────")
if models and imgs and result["success"]:
    stages = result.get("preprocessing_stages", {})
    check("4 stage keys present",
          set(stages.keys()) == {"original","constancy","clahe","segmented"})
    for k, v in stages.items():
        check(f"Stage '{k}' is non-empty BGR array",
              v is not None and v.size > 0 and v.ndim == 3)

# ── Test 5: PIL RGB→BGR conversion ───────────────────────────────────────
print("\n── Test 5: PIL RGB→BGR conversion ─────────────────────────────────")
if imgs:
    pil_rgb = Image.open(imgs[0]).convert("RGB")
    rgb_arr = np.array(pil_rgb)
    bgr_arr = _to_bgr(pil_rgb)
    check("Output shape matches input",  bgr_arr.shape == rgb_arr.shape)
    check("BGR channel 0 ≠ RGB channel 0 (channels were swapped)",
          not np.array_equal(bgr_arr[:,:,0], rgb_arr[:,:,0]))
    check("BGR channel 0 == RGB channel 2 (B = R after swap)",
          np.array_equal(bgr_arr[:,:,0], rgb_arr[:,:,2]))

# ── Test 6: EXPECTED_FEATURES constant ───────────────────────────────────
print("\n── Test 6: Feature contract constant ──────────────────────────────")
check("EXPECTED_FEATURES == 66", EXPECTED_FEATURES == 66,
      f"got {EXPECTED_FEATURES}")

# ── Test 7: Feature dimension mismatch guard ──────────────────────────────
print("\n── Test 7: Dimension mismatch guard ───────────────────────────────")
if models and imgs:
    import cv2
    import src.preprocessing as pp
    import src.feature_extraction as fe
    # Patch on src.predict (where extract_features was imported into)
    import src.predict as predict_module
    _orig = predict_module.extract_features
    predict_module.extract_features = lambda img: np.zeros(40, dtype=np.float32)  # wrong dim → triggers guard
    bad_result = predict_image(Image.open(imgs[0]), models)
    predict_module.extract_features = _orig   # restore
    check("Mismatch triggers success=False",  not bad_result["success"])
    check("Error message mentions mismatch",
          "mismatch" in (bad_result.get("error") or "").lower())

# ── Test 8: Existing modules still import cleanly ─────────────────────────
print("\n── Test 8: Existing module imports unaffected ──────────────────────")
try:
    from src.preprocessing     import preprocess_image, segment_leaf
    from src.feature_extraction import extract_features, get_feature_names
    from src.models             import load_model, train_knn
    from src.ensemble           import WeightedEnsembleClassifier, build_ensemble
    from src.evaluation         import evaluate_model, plot_confusion_matrix
    from src.utils              import set_seed, make_splits, load_splits
    check("All 6 existing src modules import", True)
except Exception as e:
    check("All 6 existing src modules import", False, str(e))

# ── Test 9: metrics_summary.csv loadable ─────────────────────────────────
print("\n── Test 9: Evaluation artifacts ───────────────────────────────────")
csv_path = os.path.join("reports", "metrics_summary.csv")
check("metrics_summary.csv exists", os.path.exists(csv_path))
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    check("CSV has required columns",
          {"model","subset","accuracy","f1_macro"}.issubset(df.columns))
    check("CSV has both 'full' and 'field' subsets",
          set(df["subset"].unique()) == {"full","field"})
    check("CSV has 4 models × 2 subsets = 8 rows", len(df) == 8,
          f"got {len(df)} rows")

figures = glob.glob("reports/figures/*.png")
check(f"At least 10 figure PNGs exist", len(figures) >= 10,
      f"found {len(figures)}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
if not errors:
    print("  All tests passed. ✓")
else:
    print(f"  {len(errors)} test(s) FAILED:")
    for e in errors:
        print(f"    • {e}")
print("="*60)
sys.exit(0 if not errors else 1)

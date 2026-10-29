"""
predict.py
==========
Reusable inference module for single-image plant disease prediction.

This module is the ONLY place that bridges an uploaded image to the
existing trained models.  It does NOT re-implement any ML logic — it
calls the existing functions from:
  src.preprocessing     → preprocess_image()
  src.feature_extraction → extract_features(), get_feature_names()
  src.models            → load_model()
  src.ensemble          → WeightedEnsembleClassifier.load()

Key design decisions (based on inspection):
  • All base models are sklearn Pipelines: [StandardScaler → classifier]
    → StandardScaler is INSIDE the pipeline.  No manual scaling here.
  • Streamlit uploads give a PIL Image (RGB order).
    → Must convert RGB→BGR before passing to preprocess_image().
  • Expected feature count: 56 (verified from scaler.n_features_in_).
    → A validation check is performed before every prediction.
  • All three models and the ensemble share the same class list in the
    same alphabetical order — no realignment needed at this level
    (the ensemble's _align_proba already handles it internally).

Public API
----------
  load_all_models(models_dir)   → dict of fitted model objects
  predict_image(pil_or_bgr, models)
      → {
          "success"       : bool,
          "error"         : str | None,
          "preprocessed"  : np.ndarray (BGR),
          "features"      : np.ndarray (56,),
          "feature_names" : list[str],
          "predictions"   : {
              "knn"      : {"label": str, "proba": dict},
              "svm"      : {"label": str, "proba": dict},
              "rf"       : {"label": str, "proba": dict},
              "ensemble" : {"label": str, "proba": dict},
          },
          "preprocessing_stages": {
              "original"   : np.ndarray (BGR),
              "constancy"  : np.ndarray (BGR),
              "clahe"      : np.ndarray (BGR),
              "segmented"  : np.ndarray (BGR),
          }
        }
"""

import os
import sys
import numpy as np

import cv2

# ── Make src importable regardless of working directory ───────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preprocessing import (
    preprocess_image,
    color_constancy,
    normalize_illumination,
    segment_leaf,
)
from src.feature_extraction import extract_features, get_feature_names
from src.models   import load_model
from src.ensemble import WeightedEnsembleClassifier

# ── Expected feature count (verified from trained models) ─────────────────
# v2 feature extraction: 20 colour + 30 texture + 16 shape = 66
EXPECTED_FEATURES = 66

# ── Default model directory ───────────────────────────────────────────────
DEFAULT_MODELS_DIR = os.path.join(_ROOT, "models")


# ===========================================================================
# Model loader  (called once; results should be cached in Streamlit)
# ===========================================================================

def load_all_models(models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    """
    Load all four trained model objects from disk.

    Returns a dict:
        {
          "knn"      : fitted sklearn Pipeline,
          "svm"      : fitted sklearn Pipeline,
          "rf"       : fitted sklearn Pipeline,
          "ensemble" : WeightedEnsembleClassifier,
        }

    Raises FileNotFoundError with a clear message if any file is missing.
    """
    paths = {
        "knn" : os.path.join(models_dir, "knn_model.pkl"),
        "svm" : os.path.join(models_dir, "svm_model.pkl"),
        "rf"  : os.path.join(models_dir, "rf_model.pkl"),
    }
    ens_path = os.path.join(models_dir, "ensemble.pkl")

    missing = [p for p in list(paths.values()) + [ens_path]
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "The following model files are missing:\n"
            + "\n".join(f"  • {p}" for p in missing)
            + "\n\nRun the training pipeline first:\n"
            + "  python run_pipeline.py"
        )

    models = {}
    for name, path in paths.items():
        models[name] = load_model(path)

    models["ensemble"] = WeightedEnsembleClassifier.load(ens_path)
    return models


# ===========================================================================
# Image conversion  (PIL RGB → BGR numpy, or pass-through for BGR arrays)
# ===========================================================================

def _to_bgr(image) -> np.ndarray:
    """
    Convert an input image to a BGR uint8 numpy array.

    Accepts:
      • PIL.Image.Image  (from Streamlit file_uploader)
      • np.ndarray in RGB order  (H×W×3, uint8)
      • np.ndarray in BGR order  (H×W×3, uint8) — returned as-is

    The function uses the PIL mode attribute to decide whether an RGB→BGR
    flip is needed, so it never double-converts.

    Parameters
    ----------
    image : PIL.Image | np.ndarray

    Returns
    -------
    np.ndarray – BGR uint8 array
    """
    try:
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            # PIL always stores as RGB; convert to BGR for OpenCV
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        pass

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Greyscale → convert to BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            # RGBA → BGR
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        # Assume RGB numpy array (e.g. from matplotlib or skimage load)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    raise TypeError(
        f"Unsupported image type: {type(image)}. "
        "Pass a PIL.Image or an RGB/BGR numpy array."
    )


# ===========================================================================
# Preprocessing stages  (for visualisation in the UI)
# ===========================================================================

def get_preprocessing_stages(image_bgr: np.ndarray,
                              target_size: tuple = (224, 224)) -> dict:
    """
    Run the preprocessing pipeline step by step and return intermediate
    BGR images so the UI can display them.

    This does NOT change what is passed to feature extraction — it only
    exposes the intermediate stages for visualisation.

    Parameters
    ----------
    image_bgr   : np.ndarray – raw BGR image (already resized to target_size)
    target_size : tuple      – resize target

    Returns
    -------
    dict:
        "original"   : resized raw BGR image
        "constancy"  : after Gray-World colour constancy
        "clahe"      : after CLAHE illumination normalisation
        "segmented"  : after leaf segmentation (final preprocessed)
    """
    # Step 1 – resize only (no other processing yet)
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)

    # Step 2 – colour constancy
    after_constancy = color_constancy(resized)

    # Step 3 – CLAHE illumination normalisation
    after_clahe = normalize_illumination(after_constancy)

    # Step 4 – leaf segmentation
    after_segment = segment_leaf(after_clahe)

    return {
        "original"  : resized,
        "constancy" : after_constancy,
        "clahe"     : after_clahe,
        "segmented" : after_segment,
    }


# ===========================================================================
# Single-model prediction helper
# ===========================================================================

def _predict_one(model, features_2d: np.ndarray) -> dict:
    """
    Run prediction for a single fitted sklearn Pipeline or
    WeightedEnsembleClassifier on a (1, 56) feature matrix.

    Returns
    -------
    dict:
        "label" : str   – predicted class name
        "proba" : dict  – {class_name: float probability} or None if unsupported
    """
    label = str(model.predict(features_2d)[0])

    proba_dict = None
    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(features_2d)[0]  # shape (n_classes,)
            classes   = model.classes_
            proba_dict = {
                str(cls): round(float(p), 6)
                for cls, p in zip(classes, proba_arr)
            }
        except Exception:
            proba_dict = None

    return {"label": label, "proba": proba_dict}


# ===========================================================================
# Main inference function
# ===========================================================================

def predict_image(
    image,
    models:      dict,
    target_size: tuple = (224, 224),
) -> dict:
    """
    Full inference pipeline for a single uploaded image.

    Parameters
    ----------
    image        : PIL.Image | np.ndarray (RGB or BGR)
                   The uploaded image from Streamlit or any other source.
    models       : dict  – output of load_all_models()
    target_size  : tuple – must match what was used during training (224×224)

    Returns
    -------
    dict:
        success            : bool
        error              : str | None   (set when success is False)
        preprocessed       : np.ndarray (BGR, 224×224) – the image fed to features
        features           : np.ndarray (56,)
        feature_names      : list[str] of length 56
        predictions        : dict with keys "knn", "svm", "rf", "ensemble"
                             each → {"label": str, "proba": dict | None}
        preprocessing_stages : dict with keys "original","constancy","clahe","segmented"
    """
    result = {
        "success"              : False,
        "error"                : None,
        "preprocessed"         : None,
        "features"             : None,
        "feature_names"        : None,
        "predictions"          : None,
        "preprocessing_stages" : None,
    }

    # ── Step 1 – convert to BGR numpy array ──────────────────────────────
    try:
        bgr = _to_bgr(image)
    except Exception as exc:
        result["error"] = f"Image conversion failed: {exc}"
        return result

    if bgr is None or bgr.size == 0:
        result["error"] = "Image is empty after conversion."
        return result

    # ── Step 2 – capture preprocessing stages (for visualisation) ─────────
    try:
        stages = get_preprocessing_stages(bgr, target_size=target_size)
        result["preprocessing_stages"] = stages
    except Exception as exc:
        # Non-fatal; proceed without visualisation stages
        result["preprocessing_stages"] = {}

    # ── Step 3 – full preprocessing (same pipeline used during training) ──
    try:
        preprocessed = preprocess_image(bgr, target_size=target_size)
    except Exception as exc:
        result["error"] = f"Preprocessing failed: {exc}"
        return result

    if preprocessed is None or preprocessed.size == 0:
        result["error"] = "Preprocessing returned an empty image."
        return result

    result["preprocessed"] = preprocessed

    # ── Step 4 – feature extraction ───────────────────────────────────────
    try:
        features = extract_features(preprocessed)   # → (56,) float32
    except Exception as exc:
        result["error"] = f"Feature extraction failed: {exc}"
        return result

    # ── Step 5 – feature dimension validation ─────────────────────────────
    actual_count = features.shape[0]
    if actual_count != EXPECTED_FEATURES:
        result["error"] = (
            f"Feature dimension mismatch: "
            f"expected {EXPECTED_FEATURES}, got {actual_count}. "
            "The feature extraction pipeline and the trained models are "
            "incompatible. Check that src/feature_extraction.py has not "
            "been modified after training."
        )
        return result

    result["features"]      = features
    result["feature_names"] = get_feature_names()

    # ── Step 6 – predict with each model ──────────────────────────────────
    # Reshape to (1, 56) for sklearn
    X = features.reshape(1, -1)

    predictions = {}
    for name in ["knn", "svm", "rf", "ensemble"]:
        model = models.get(name)
        if model is None:
            predictions[name] = {"label": "Model not loaded", "proba": None}
            continue
        try:
            predictions[name] = _predict_one(model, X)
        except Exception as exc:
            predictions[name] = {
                "label": f"Prediction error: {exc}",
                "proba": None,
            }

    result["predictions"] = predictions
    result["success"]     = True
    return result


# ===========================================================================
# Smoke test  (python src/predict.py <image_path>)
# ===========================================================================

if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        # Use first available demo image
        import glob
        imgs = glob.glob(os.path.join(_ROOT, "data", "plantvillage",
                                      "**", "*.jpg"), recursive=True)
        if not imgs:
            print("Usage: python src/predict.py <image_path>")
            sys.exit(1)
        img_path = imgs[0]
        print(f"No path given — using: {img_path}")
    else:
        img_path = sys.argv[1]

    print("\nLoading models …")
    try:
        models = load_all_models()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("Running inference …")
    from PIL import Image as PILImage
    pil_img = PILImage.open(img_path)
    out = predict_image(pil_img, models)

    if not out["success"]:
        print(f"\nINFERENCE FAILED: {out['error']}")
        sys.exit(1)

    print(f"\nFeature vector : shape={out['features'].shape}, "
          f"min={out['features'].min():.4f}, max={out['features'].max():.4f}")

    print("\nPredictions:")
    for model_name, pred in out["predictions"].items():
        label = pred["label"]
        if pred["proba"]:
            top_p = max(pred["proba"].values())
            print(f"  {model_name.upper():<10s}: {label}  "
                  f"(top prob={top_p:.4f})")
        else:
            print(f"  {model_name.upper():<10s}: {label}")

    print("\nSmoke test passed.")

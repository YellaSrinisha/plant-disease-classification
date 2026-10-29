"""
feature_extraction.py  (v2 — 66 features)
==========================================
Multi-family feature engineering for plant disease classification.

Feature families
----------------
  1. COLOR   (20) – RGB/HSV channel stats + vegetation indices
  2. TEXTURE (30) – GLCM properties + LBP histogram
  3. SHAPE   (16) – normalised contour descriptors + Hu moments + lesion stats

Total: 66 features  (was 56 — added 10 shape/lesion features)

Key changes from v1
-------------------
• Shape features are now scale-invariant:
    - area / image_area          (replaces raw pixel count)
    - perimeter / image_perimeter (replaces raw pixel length)
    - compactness, eccentricity, aspect_ratio, solidity unchanged
• Added 7 Hu moments (log-scaled) — rotation, scale, and translation invariant
• Added 3 lesion-level features:
    - lesion_count      number of significant dark/brown spots on the leaf
    - lesion_density    lesion_count / leaf_area
    - lesion_area_ratio total lesion area / leaf area
• get_feature_names() updated to return 66 names

Public API
----------
  extract_features(image_bgr)            → 1-D float32 array, length 66
  extract_features_from_dataset(df, ...) → X, y, skipped
  get_feature_names()                    → list of 66 strings
"""

import os
import warnings
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src.preprocessing import load_image, convert_color_spaces

warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# FEATURE FAMILY 1 – COLOR  (20 features, unchanged from v1)
# ===========================================================================

def extract_color_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Colour statistics and vegetation indices.

    Features (20)
    -------------
    RGB stats  : mean/std of R, G, B                     (6)
    HSV stats  : mean/std of H, S, V                     (6)
    Ratios on foreground pixels only:
        G/R  mean & std                                   (2)
        G/B  mean & std                                   (2)
        NDI = (G−R)/(G+R) mean & std                     (2)
        ExG = 2G−R−B mean & std                          (2)
    """
    features = []

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    for ch in range(3):
        features.extend([rgb[:, :, ch].mean(), rgb[:, :, ch].std()])

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    for ch in range(3):
        features.extend([hsv[:, :, ch].mean(), hsv[:, :, ch].std()])

    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    fg_mask = gray > 0
    r_px = rgb[:, :, 0][fg_mask].astype(np.float32) + 1e-6
    g_px = rgb[:, :, 1][fg_mask].astype(np.float32) + 1e-6
    b_px = rgb[:, :, 2][fg_mask].astype(np.float32) + 1e-6

    if len(g_px) == 0:
        features.extend([0.0] * 8)
    else:
        for ratio in [g_px / r_px,
                      g_px / b_px,
                      (g_px - r_px) / (g_px + r_px),
                      2 * g_px - r_px - b_px]:
            features.extend([ratio.mean(), ratio.std()])

    return np.array(features, dtype=np.float32)   # (20,)


# ===========================================================================
# FEATURE FAMILY 2 – TEXTURE  (30 features, unchanged from v1)
# ===========================================================================

def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    """
    GLCM texture properties averaged over 4 directions.
    Returns 4 float32 values: contrast, correlation, energy, homogeneity.
    """
    gray_r  = (gray // 4).astype(np.uint8)
    angles  = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm    = graycomatrix(gray_r, distances=[1], angles=angles,
                           levels=64, symmetric=True, normed=True)
    return np.array(
        [float(graycoprops(glcm, p).mean())
         for p in ["contrast", "correlation", "energy", "homogeneity"]],
        dtype=np.float32
    )


def extract_lbp_features(gray: np.ndarray,
                          n_points: int = 24,
                          radius:   int = 3,
                          n_bins:   int = 26) -> np.ndarray:
    """
    Normalised LBP histogram (26 bins) computed on foreground pixels only.
    """
    lbp     = local_binary_pattern(gray, n_points, radius, method="uniform")
    fg_mask = gray > 0
    fg_lbp  = lbp[fg_mask] if fg_mask.any() else lbp.ravel()
    hist, _ = np.histogram(fg_lbp, bins=n_bins,
                           range=(0, n_points + 2), density=False)
    hist    = hist.astype(np.float32)
    hist   /= (hist.sum() + 1e-6)
    return hist   # (26,)


def extract_texture_features(image_bgr: np.ndarray) -> np.ndarray:
    """Combine GLCM (4) + LBP (26) = 30 texture features."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return np.concatenate([extract_glcm_features(gray),
                           extract_lbp_features(gray)])   # (30,)


# ===========================================================================
# FEATURE FAMILY 3 – SHAPE  (16 features — upgraded from 6)
# ===========================================================================

def extract_shape_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Scale-invariant geometric descriptors + Hu moments + lesion stats.

    Features (16)
    -------------
    Normalised area       = contour_area / image_area              (1)
    Normalised perimeter  = perimeter / image_perimeter            (1)
    Compactness           = 4π·area / perimeter²                   (1)
    Eccentricity          = minor_axis / major_axis (0–1)          (1)
    Aspect ratio          = bounding_box_w / h                     (1)
    Solidity              = area / convex_hull_area                 (1)
    Hu moments (log)      = –sign(h)·log10(|h|+ε) for h0…h6       (7)
    Lesion count          = number of dark/brown spots             (1)
    Lesion density        = lesion_count / (leaf_area + ε)         (1)
    Lesion area ratio     = total_lesion_area / (leaf_area + ε)    (1)

    Returns zeros on failure (no contour found).
    """
    h_img, w_img = image_bgr.shape[:2]
    img_area      = float(h_img * w_img)
    img_perim     = float(2 * (h_img + w_img))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(16, dtype=np.float32)

    contour   = max(contours, key=cv2.contourArea)
    area_raw  = float(cv2.contourArea(contour))
    if area_raw < 1:
        return np.zeros(16, dtype=np.float32)

    perim_raw   = float(cv2.arcLength(contour, True))
    compactness = 4 * np.pi * area_raw / (perim_raw ** 2 + 1e-6)

    # Eccentricity from fitted ellipse
    eccentricity = 0.0
    if len(contour) >= 5:
        _, (minor, major), _ = cv2.fitEllipse(contour)
        eccentricity = float(minor / (major + 1e-6))

    # Aspect ratio from bounding box
    _, _, bw, bh = cv2.boundingRect(contour)
    aspect_ratio  = float(bw / (bh + 1e-6))

    # Solidity
    hull         = cv2.convexHull(contour)
    hull_area    = float(cv2.contourArea(hull))
    solidity     = float(area_raw / (hull_area + 1e-6))

    # ── Hu moments (7 values) — scale/rotation/translation invariant ──────
    moments = cv2.moments(contour)
    hu_raw  = cv2.HuMoments(moments).flatten()   # shape (7,)
    # Log-scale transform: preserves sign, compresses dynamic range
    hu_log  = np.array(
        [-np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu_raw],
        dtype=np.float32
    )

    # ── Lesion features ───────────────────────────────────────────────────
    # Build leaf mask (foreground of the segmented image)
    leaf_mask    = binary                         # already computed above
    leaf_area_px = float(np.count_nonzero(leaf_mask))

    # Detect lesions: dark-brown/yellow spots using HSV
    # Brown lesions: low saturation + low value, or hue 10-30
    # Yellow lesions: hue 20-40, high saturation
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lesion_mask_brown = cv2.inRange(
        hsv, np.array([0,  30, 10]), np.array([30, 200, 160])
    )
    lesion_mask_yellow = cv2.inRange(
        hsv, np.array([15, 80, 80]), np.array([40, 255, 255])
    )
    lesion_mask = cv2.bitwise_or(lesion_mask_brown, lesion_mask_yellow)

    # Restrict lesions to the leaf region
    if leaf_area_px > 0:
        lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask,
                                       mask=leaf_mask)

    # Morphological cleaning of lesion mask
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN,  k_small)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, k_small)

    # Count individual lesion blobs
    n_labels, _, l_stats, _ = cv2.connectedComponentsWithStats(
        lesion_mask, connectivity=8
    )
    # Filter tiny blobs (< 0.1% of image area)
    min_lesion_px = max(4, int(img_area * 0.001))
    lesion_blobs  = [l_stats[i, cv2.CC_STAT_AREA]
                     for i in range(1, n_labels)
                     if l_stats[i, cv2.CC_STAT_AREA] >= min_lesion_px]

    lesion_count      = float(len(lesion_blobs))
    total_lesion_area = float(sum(lesion_blobs))
    lesion_density    = lesion_count    / (leaf_area_px + 1e-6)
    lesion_area_ratio = total_lesion_area / (leaf_area_px + 1e-6)

    # ── Assemble all 16 shape features ────────────────────────────────────
    basic = np.array([
        area_raw  / img_area,          # normalised area
        perim_raw / img_perim,         # normalised perimeter
        compactness,
        eccentricity,
        aspect_ratio,
        solidity,
    ], dtype=np.float32)

    lesion = np.array([
        lesion_count,
        lesion_density,
        lesion_area_ratio,
    ], dtype=np.float32)

    return np.concatenate([basic, hu_log, lesion])   # (6 + 7 + 3) = (16,)


# ===========================================================================
# COMBINED FEATURE VECTOR  (66 features)
# ===========================================================================

def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract and concatenate all three feature families.

    Feature counts
    --------------
    Color   : 20
    Texture : 30
    Shape   : 16  (was 6 in v1)
    Total   : 66

    Parameters
    ----------
    image_bgr : np.ndarray – preprocessed BGR uint8 image

    Returns
    -------
    np.ndarray – 1-D float32 array, length 66
    """
    color   = extract_color_features(image_bgr)     # (20,)
    texture = extract_texture_features(image_bgr)   # (30,)
    shape   = extract_shape_features(image_bgr)     # (16,)
    return np.concatenate([color, texture, shape])  # (66,)


# ===========================================================================
# DATASET-LEVEL EXTRACTION
# ===========================================================================

def extract_features_from_dataset(
    df:            pd.DataFrame,
    target_size:   tuple = (224, 224),
    show_progress: bool  = True
) -> tuple:
    """
    Load every image in df, preprocess, and extract the 66-feature vector.
    Skips unreadable / corrupted images gracefully.

    Parameters
    ----------
    df            : pd.DataFrame – must have 'file_path' and 'label' columns
    target_size   : tuple        – passed to load_image()
    show_progress : bool         – tqdm progress bar

    Returns
    -------
    X       : np.ndarray  shape (n, 66), float32
    y       : np.ndarray  shape (n,),    labels
    skipped : list        file paths that failed
    """
    feature_list, label_list, skipped = [], [], []

    iterator = tqdm(df.iterrows(), total=len(df),
                    desc="Extracting features",
                    disable=not show_progress)

    for _, row in iterator:
        path  = row["file_path"]
        label = row["label"]
        img   = load_image(path, target_size=target_size)
        if img is None:
            skipped.append(path)
            continue
        try:
            feats = extract_features(img)
            feature_list.append(feats)
            label_list.append(label)
        except Exception as exc:
            print(f"[WARNING] Feature extraction failed for {path}: {exc}")
            skipped.append(path)

    if not feature_list:
        raise RuntimeError(
            "No features extracted. Check data paths and image files."
        )

    X = np.stack(feature_list, axis=0).astype(np.float32)
    y = np.array(label_list)

    if skipped:
        print(f"\n[INFO] Skipped {len(skipped)} image(s).")

    return X, y, skipped


# ===========================================================================
# FEATURE NAMES  (66 names)
# ===========================================================================

def get_feature_names() -> list:
    """Return ordered list of 66 feature names (matches extract_features output)."""
    names = []

    # Color (20)
    for ch in ["R", "G", "B"]:
        names += [f"color_mean_{ch}", f"color_std_{ch}"]
    for ch in ["H", "S", "V"]:
        names += [f"color_mean_{ch}", f"color_std_{ch}"]
    names += ["ratio_GR_mean", "ratio_GR_std",
              "ratio_GB_mean", "ratio_GB_std",
              "ndi_mean",      "ndi_std",
              "exg_mean",      "exg_std"]

    # Texture (30)
    names += ["glcm_contrast", "glcm_correlation",
              "glcm_energy",   "glcm_homogeneity"]
    names += [f"lbp_bin_{i}" for i in range(26)]

    # Shape (16)
    names += ["shape_area_norm", "shape_perim_norm",
              "shape_compactness", "shape_eccentricity",
              "shape_aspect_ratio", "shape_solidity"]
    names += [f"hu_moment_{i}" for i in range(7)]
    names += ["lesion_count", "lesion_density", "lesion_area_ratio"]

    assert len(names) == 66, f"Expected 66 names, got {len(names)}"
    return names


# ===========================================================================
# Smoke-test
# ===========================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <image_path>")
        sys.exit(0)

    img = load_image(sys.argv[1])
    if img is not None:
        feats = extract_features(img)
        names = get_feature_names()
        print(f"\nExtracted {len(feats)} features:")
        for n, v in zip(names, feats):
            print(f"  {n:<35s} = {v:.6f}")
    else:
        print("Failed to load / preprocess image.")

"""
preprocessing.py  (v2 — field-robust)
======================================
Image preprocessing pipeline for plant disease classification.

Steps applied to every image (in order):
  1. Resize              – fixed 224×224 for consistent feature extraction
  2. Color constancy     – Gray-World + per-channel histogram stretching
  3. Illumination norm   – CLAHE on L* channel of Lab*
  4. Leaf segmentation   – multi-strategy with graceful fallback

Key change in v2: segment_leaf() now uses THREE strategies and only applies
a mask when segmentation produces a meaningful foreground region (≥5% of
pixels). On complex backgrounds where all strategies fail, the full image is
used rather than producing a corrupt near-empty image. This is the primary
fix for the lab-to-field accuracy gap.

All functions accept and return BGR numpy arrays (OpenCV convention).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Color-space conversion helpers
# ---------------------------------------------------------------------------

def convert_color_spaces(image_bgr: np.ndarray) -> dict:
    """
    Convert a BGR image to several colour spaces.

    Returns
    -------
    dict with keys: 'bgr', 'rgb', 'hsv', 'lab', 'gray'
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("convert_color_spaces: received an empty image.")
    return {
        "bgr":  image_bgr,
        "rgb":  cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "hsv":  cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV),
        "lab":  cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab),
        "gray": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
    }


# ---------------------------------------------------------------------------
# 2. Color constancy  (Gray-World + histogram stretching)
# ---------------------------------------------------------------------------

def color_constancy(image_bgr: np.ndarray) -> np.ndarray:
    """
    Two-stage colour constancy:
      a) Gray-World: scale each channel so its mean equals the overall mean.
      b) Per-channel histogram stretching to [5th, 95th] percentile range,
         which handles the common case where Gray-World over-corrects green
         dominant scenes (field leaf photos).

    Parameters
    ----------
    image_bgr : np.ndarray – input BGR uint8 image

    Returns
    -------
    np.ndarray – white-balanced BGR image (uint8)
    """
    img = image_bgr.astype(np.float32)

    # Stage a — Gray-World
    means   = [img[:, :, c].mean() for c in range(3)]
    overall = sum(means) / 3.0
    for c in range(3):
        if means[c] > 1e-5:
            img[:, :, c] *= (overall / means[c])
    img = np.clip(img, 0, 255)

    # Stage b — per-channel percentile stretch
    for c in range(3):
        lo = float(np.percentile(img[:, :, c], 2))
        hi = float(np.percentile(img[:, :, c], 98))
        if hi - lo > 10:                        # only stretch if range is meaningful
            img[:, :, c] = (img[:, :, c] - lo) / (hi - lo) * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Illumination normalisation  (CLAHE on L* channel)
# ---------------------------------------------------------------------------

def normalize_illumination(image_bgr: np.ndarray,
                            clip_limit: float = 2.5,
                            tile_grid:  tuple  = (8, 8)) -> np.ndarray:
    """
    Reduce uneven lighting by applying CLAHE to the Lightness channel of Lab*.

    clip_limit raised to 2.5 (from 2.0) for stronger contrast enhancement
    on field images which tend to have more illumination variation.

    Parameters
    ----------
    image_bgr  : np.ndarray – input BGR uint8 image
    clip_limit : float      – CLAHE threshold (default 2.5)
    tile_grid  : tuple      – tile grid (default 8×8)

    Returns
    -------
    np.ndarray – illumination-normalised BGR image (uint8)
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_Lab2BGR)


# ---------------------------------------------------------------------------
# 4. Leaf segmentation  (multi-strategy with fallback)
# ---------------------------------------------------------------------------

# Minimum fraction of image pixels the mask must cover to be trusted.
# Below this threshold we fall back to the full image.
_MIN_MASK_FILL = 0.05   # 5 %
_MAX_MASK_FILL = 0.97   # 97% — avoid masks that swallow the whole image


def _apply_morphology(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Close holes then open small noise on a binary mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def _mask_fill_ratio(mask: np.ndarray) -> float:
    total = mask.shape[0] * mask.shape[1]
    return float(np.count_nonzero(mask)) / total if total > 0 else 0.0


def _strategy_exg_sat(image_bgr: np.ndarray) -> np.ndarray:
    """
    Strategy 1 (original): Excess Green index + HSV Saturation Otsu.
    Works well for clean green leaves on uniform backgrounds.
    """
    img_f = image_bgr.astype(np.float32) + 1e-6
    exg   = 2.0 * img_f[:, :, 1] - img_f[:, :, 0] - img_f[:, :, 2]
    exg_n = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m_exg = cv2.threshold(exg_n, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, m_sat = cv2.threshold(hsv[:, :, 1], 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(m_exg, m_sat)
    mask = _apply_morphology(mask, kernel_size=15)
    return _largest_component(mask)


def _strategy_hsv_green(image_bgr: np.ndarray) -> np.ndarray:
    """
    Strategy 2: HSV green-range thresholding.
    Catches yellow-green diseased leaves and brownish lesions better than ExG.
    Hue range 25–95 covers yellow-green to green.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Lower range: yellow-green
    m1  = cv2.inRange(hsv, np.array([25,  30,  30]),
                           np.array([95, 255, 255]))
    # Also capture brownish/dark-green diseased areas via value
    m2  = cv2.inRange(hsv, np.array([0, 40, 20]),
                           np.array([30, 255, 200]))
    mask = cv2.bitwise_or(m1, m2)
    mask = _apply_morphology(mask, kernel_size=13)
    return _largest_component(mask)


def _strategy_center_bias(image_bgr: np.ndarray) -> np.ndarray:
    """
    Strategy 3: Assume the leaf is roughly centred in the frame.
    Builds a Gaussian-weighted saliency map on the greyscale image,
    thresholds it, and returns the largest component.

    This is the last-resort fallback for complex field photos where
    colour-based segmentation fails entirely.
    """
    h, w   = image_bgr.shape[:2]
    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Gaussian centre-bias weight
    cx, cy = w / 2, h / 2
    sigma  = min(w, h) * 0.35
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    weight = np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2))

    saliency = (gray.astype(np.float32) * weight)
    saliency = cv2.normalize(saliency, None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(saliency, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _apply_morphology(mask, kernel_size=11)
    return _largest_component(mask)


def segment_leaf(image_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the leaf from the background using a cascade of three strategies.

    The cascade tries each strategy in order and accepts the first mask whose
    fill ratio falls in [_MIN_MASK_FILL, _MAX_MASK_FILL].  If all strategies
    fail, the full (unsegmented) image is returned rather than a near-empty
    black image — this is the critical fix for complex field photos.

    Strategy order
    --------------
    1. ExG + HSV Saturation Otsu  (fast, works on clean lab images)
    2. HSV green-range threshold  (handles yellow/brown diseased leaves)
    3. Centre-bias Gaussian       (last resort for complex backgrounds)
    4. Full image fallback        (never corrupt the features)

    Parameters
    ----------
    image_bgr : np.ndarray – input BGR uint8 image (after resize + constancy)

    Returns
    -------
    np.ndarray – BGR image; background zeroed if segmentation succeeded,
                 original image if all strategies failed
    """
    strategies = [
        _strategy_exg_sat,
        _strategy_hsv_green,
        _strategy_center_bias,
    ]

    for strategy in strategies:
        try:
            mask  = strategy(image_bgr)
            ratio = _mask_fill_ratio(mask)
            if _MIN_MASK_FILL <= ratio <= _MAX_MASK_FILL:
                return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        except Exception:
            continue   # silently try next strategy

    # Fallback: return the full image — better than returning a black frame
    return image_bgr.copy()


# ---------------------------------------------------------------------------
# 5. Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_image(image_bgr: np.ndarray,
                     target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. Resize             – 224×224
      2. Color constancy    – Gray-World + percentile stretch
      3. Illumination norm  – CLAHE (L* channel, clip=2.5)
      4. Leaf segmentation  – 3-strategy cascade with fallback

    Parameters
    ----------
    image_bgr   : np.ndarray – raw BGR image
    target_size : tuple      – (width, height), default 224×224

    Returns
    -------
    np.ndarray – preprocessed BGR uint8 image
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("preprocess_image: received a None or empty image.")

    img = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    img = color_constancy(img)
    img = normalize_illumination(img)
    img = segment_leaf(img)
    return img


# ---------------------------------------------------------------------------
# 6. Safe image loader
# ---------------------------------------------------------------------------

def load_image(file_path: str,
               target_size: tuple = (224, 224)) -> "np.ndarray | None":
    """
    Load an image from disk and run the full preprocessing pipeline.
    Returns None on any error.
    """
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"[WARNING] Could not read image: {file_path}")
            return None
        return preprocess_image(img, target_size=target_size)
    except Exception as exc:
        print(f"[WARNING] Error processing {file_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path>")
        sys.exit(0)
    result = load_image(sys.argv[1])
    if result is not None:
        out = os.path.join(os.path.dirname(sys.argv[1]), "preprocessed_v2.jpg")
        cv2.imwrite(out, result)
        fg = (cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) > 0).mean()
        print(f"Saved: {out} | foreground={fg*100:.1f}%")
    else:
        print("Preprocessing failed.")

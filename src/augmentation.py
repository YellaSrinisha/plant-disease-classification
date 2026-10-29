"""
augmentation.py  (v3 — domain-aware field augmentation)
=========================================================
Training-time data augmentation designed to simulate the conditions a farmer
encounters when photographing leaves.

v3 changes
----------
• augment_image() now accepts a `source` parameter ('plantvillage' or 'plantdoc').
  - PlantVillage images  → STRONG field-simulation transforms (they need to look
    more like real farm photos to close the domain gap).
  - PlantDoc images      → LIGHT augmentation only (they are already field images;
    over-augmenting them destroys the real-world signal we want to learn).
• Stronger JPEG compression for PlantVillage: quality 10-45 (was 30-75).
• Stronger hue shift for PlantVillage: +-40 (was +-20).
• Stronger brightness delta for PlantVillage: +-0.9 (was +-0.75).
• Perspective warp max_shift raised to 0.15 for PlantVillage (was 0.08).
• Added coarse dropout (random black rectangles) to PlantVillage — simulates
  partial occlusion by other leaves or dirt, very common in field photos.
• augment_dataset_files() now accepts a `sources` list and passes it through.

Public API
----------
  augment_image(image_bgr, seed, source)  -> augmented BGR image
  augment_dataset_files(paths, labels,
                        n_augments, seed,
                        sources)          -> aug_paths, aug_labels
  cleanup_augmented_temp()
"""

import cv2
import numpy as np
import os
import shutil


# ── Individual augmentation functions ────────────────────────────────────

def random_rotation(image: np.ndarray, max_angle: float = 40.0,
                    rng=None) -> np.ndarray:
    """Rotate +-max_angle degrees with reflection border fill."""
    if rng is None: rng = np.random.default_rng()
    angle = rng.uniform(-max_angle, max_angle)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def random_brightness(image: np.ndarray, max_delta: float = 0.75,
                      rng=None) -> np.ndarray:
    """Simulate different daylight levels."""
    if rng is None: rng = np.random.default_rng()
    factor = rng.uniform(1 - max_delta, 1 + max_delta)
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def random_contrast(image: np.ndarray, max_delta: float = 0.35,
                    rng=None) -> np.ndarray:
    """Adjust image contrast around the global mean."""
    if rng is None: rng = np.random.default_rng()
    factor = rng.uniform(1 - max_delta, 1 + max_delta)
    mean   = image.mean()
    return np.clip(mean + factor * (image.astype(np.float32) - mean),
                   0, 255).astype(np.uint8)


def gaussian_noise(image: np.ndarray, std: float = 12.0,
                   rng=None) -> np.ndarray:
    """Add sensor-style Gaussian noise."""
    if rng is None: rng = np.random.default_rng()
    noise = rng.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_pepper_noise(image: np.ndarray, prob: float = 0.02,
                      rng=None) -> np.ndarray:
    """Add salt-and-pepper noise to simulate dirty camera sensors."""
    if rng is None: rng = np.random.default_rng()
    out  = image.copy()
    mask = rng.random(image.shape[:2])
    out[mask < prob / 2]      = 0
    out[mask > 1 - prob / 2]  = 255
    return out


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def vertical_flip(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 0)


def hue_shift(image: np.ndarray, max_shift: int = 20,
              rng=None) -> np.ndarray:
    """Shift the Hue channel — simulates different plant varieties / light colour."""
    if rng is None: rng = np.random.default_rng()
    shift = int(rng.integers(-max_shift, max_shift))
    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                        cv2.COLOR_HSV2BGR)


def saturation_shift(image: np.ndarray, max_delta: float = 0.4,
                     rng=None) -> np.ndarray:
    """Scale saturation — simulates over-watered/dry leaves."""
    if rng is None: rng = np.random.default_rng()
    factor = rng.uniform(1 - max_delta, 1 + max_delta)
    hsv    = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def jpeg_compression(image: np.ndarray, quality_range=(30, 75),
                     rng=None) -> np.ndarray:
    """
    Simulate JPEG compression artifacts from low-quality phone photos.
    Re-encodes the image at a random quality level and decodes back.
    """
    if rng is None: rng = np.random.default_rng()
    quality = int(rng.integers(quality_range[0], quality_range[1]))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def motion_blur(image: np.ndarray, max_kernel: int = 9,
                rng=None) -> np.ndarray:
    """
    Simulate camera shake / motion blur from hand-held phone photography.
    Applies a random-direction motion blur kernel.
    """
    if rng is None: rng = np.random.default_rng()
    k = int(rng.choice([3, 5, 7, max_kernel]))
    angle = rng.uniform(0, 180)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    if kernel.sum() > 0:
        kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def shadow_strip(image: np.ndarray, rng=None) -> np.ndarray:
    """
    Add a random dark strip across the image to simulate tree/fence shadows.
    """
    if rng is None: rng = np.random.default_rng()
    h, w  = image.shape[:2]
    out   = image.astype(np.float32)
    axis  = int(rng.integers(0, 2))
    start = int(rng.integers(0, int(0.6 * (h if axis == 0 else w))))
    width = int(rng.integers(int(0.05 * h), int(0.35 * h)))
    alpha = rng.uniform(0.3, 0.7)
    if axis == 0:
        out[start:start + width, :, :] *= alpha
    else:
        out[:, start:start + width, :] *= alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def random_crop_zoom(image: np.ndarray, min_crop: float = 0.75,
                     rng=None) -> np.ndarray:
    """
    Randomly crop a subregion and resize back — simulates different distances.
    """
    if rng is None: rng = np.random.default_rng()
    h, w    = image.shape[:2]
    crop_f  = rng.uniform(min_crop, 1.0)
    ch, cw  = int(h * crop_f), int(w * crop_f)
    y0      = int(rng.integers(0, h - ch + 1))
    x0      = int(rng.integers(0, w - cw + 1))
    cropped = image[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def perspective_warp(image: np.ndarray, max_shift: float = 0.08,
                     rng=None) -> np.ndarray:
    """
    Apply a mild random perspective warp to simulate different camera angles.
    """
    if rng is None: rng = np.random.default_rng()
    h, w = image.shape[:2]
    d    = max_shift

    def pt(base_x, base_y):
        return [base_x + rng.uniform(-d, d) * w,
                base_y + rng.uniform(-d, d) * h]

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([pt(0, 0), pt(w, 0), pt(w, h), pt(0, h)])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT)


def coarse_dropout(image: np.ndarray, n_holes: int = 4,
                   max_hole_size: float = 0.12, rng=None) -> np.ndarray:
    """
    Randomly zero out small rectangular regions (black patches).

    Simulates partial leaf occlusion by other leaves, dirt, or fingers —
    very common in real field photos. Forces the model to classify from
    a partially obscured leaf, increasing robustness.
    """
    if rng is None: rng = np.random.default_rng()
    h, w = image.shape[:2]
    out  = image.copy()
    for _ in range(int(rng.integers(1, n_holes + 1))):
        hw = int(rng.uniform(0.04, max_hole_size) * h)
        ww = int(rng.uniform(0.04, max_hole_size) * w)
        y0 = int(rng.integers(0, max(1, h - hw)))
        x0 = int(rng.integers(0, max(1, w - ww)))
        out[y0:y0 + hw, x0:x0 + ww] = 0
    return out


# ── Combined augmentation (domain-aware) ─────────────────────────────────

def augment_image(image_bgr: np.ndarray,
                  seed:   int = None,
                  source: str = "plantvillage") -> np.ndarray:
    """
    Apply a random combination of field-simulating augmentations.

    v3: domain-aware.
    - source='plantvillage' → STRONG transforms to bridge the domain gap.
    - source='plantdoc'     → LIGHT transforms to preserve real-world signal.

    Parameters
    ----------
    image_bgr : np.ndarray – BGR uint8 image
    seed      : int | None – random seed for reproducibility
    source    : str        – 'plantvillage' (strong) or 'plantdoc' (light)

    Returns
    -------
    np.ndarray – augmented BGR uint8 image (same spatial size)
    """
    rng    = np.random.default_rng(seed)
    img    = image_bgr.copy()
    is_lab = (source == "plantvillage")

    # ── Geometric ─────────────────────────────────────────────────────────
    if rng.random() < 0.70:
        img = random_rotation(img, max_angle=40, rng=rng)

    if rng.random() < (0.60 if is_lab else 0.30):
        img = random_crop_zoom(img, min_crop=0.65 if is_lab else 0.80, rng=rng)

    if rng.random() < (0.55 if is_lab else 0.25):
        img = perspective_warp(img, max_shift=0.15 if is_lab else 0.06, rng=rng)

    # ── Flips ─────────────────────────────────────────────────────────────
    if rng.random() < 0.50:
        img = horizontal_flip(img)
    if rng.random() < 0.30:
        img = vertical_flip(img)

    # ── Lighting ──────────────────────────────────────────────────────────
    if rng.random() < (0.90 if is_lab else 0.50):
        img = random_brightness(img, max_delta=0.90 if is_lab else 0.40, rng=rng)

    if rng.random() < (0.70 if is_lab else 0.40):
        img = random_contrast(img, max_delta=0.45 if is_lab else 0.25, rng=rng)

    if rng.random() < (0.65 if is_lab else 0.30):
        img = shadow_strip(img, rng=rng)
        if is_lab and rng.random() < 0.35:
            img = shadow_strip(img, rng=rng)  # occasional double shadow

    # ── Colour ────────────────────────────────────────────────────────────
    if rng.random() < (0.75 if is_lab else 0.35):
        img = hue_shift(img, max_shift=40 if is_lab else 20, rng=rng)

    if rng.random() < (0.55 if is_lab else 0.30):
        img = saturation_shift(img, max_delta=0.55 if is_lab else 0.30, rng=rng)

    # ── Sensor / device artefacts ─────────────────────────────────────────
    if rng.random() < (0.65 if is_lab else 0.30):
        img = jpeg_compression(
            img,
            quality_range=(10, 45) if is_lab else (50, 85),
            rng=rng,
        )

    if rng.random() < (0.45 if is_lab else 0.25):
        img = motion_blur(img, max_kernel=11 if is_lab else 7, rng=rng)

    if rng.random() < (0.45 if is_lab else 0.20):
        img = gaussian_noise(img, std=18.0 if is_lab else 8.0, rng=rng)

    if rng.random() < (0.25 if is_lab else 0.10):
        img = salt_pepper_noise(img, prob=0.03, rng=rng)

    # ── Occlusion (lab only) ──────────────────────────────────────────────
    if is_lab and rng.random() < 0.40:
        img = coarse_dropout(img, n_holes=4, max_hole_size=0.12, rng=rng)

    return img


# ── Dataset-level augmentation ────────────────────────────────────────────

def augment_dataset_files(
    image_paths: list,
    labels:      list,
    n_augments:  int  = 5,
    seed:        int  = 42,
    sources:     list = None,
) -> tuple:
    """
    Generate augmented copies of images on disk.

    v3: accepts a `sources` list so domain-aware augmentation is applied
    per image. If sources is None, defaults all images to 'plantvillage'.

    Parameters
    ----------
    image_paths : list[str]        – original image file paths
    labels      : list[str]        – corresponding class labels
    n_augments  : int              – augmented copies per original (default 5)
    seed        : int              – base seed for reproducibility
    sources     : list[str] | None – 'plantvillage' or 'plantdoc' per image

    Returns
    -------
    aug_paths  : list[str]  – originals + augmented file paths
    aug_labels : list[str]  – corresponding labels
    """
    if sources is None:
        sources = ["plantvillage"] * len(image_paths)

    aug_paths  = list(image_paths)
    aug_labels = list(labels)

    aug_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "augmented_temp"
    )
    os.makedirs(aug_dir, exist_ok=True)

    rng  = np.random.default_rng(seed)
    idx  = 0
    n_pv = sum(1 for s in sources if s == "plantvillage")
    n_pd = sum(1 for s in sources if s == "plantdoc")
    print(f"  [Augmentation] PlantVillage: {n_pv} (strong aug)  "
          f"PlantDoc: {n_pd} (light aug)")

    for path, label, source in zip(image_paths, labels, sources):
        img = cv2.imread(path)
        if img is None:
            continue
        for _ in range(n_augments):
            aug   = augment_image(img,
                                  seed=int(rng.integers(0, 999999)),
                                  source=source)
            fname = os.path.join(aug_dir, f"aug_{idx:07d}.jpg")
            cv2.imwrite(fname, aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            aug_paths.append(fname)
            aug_labels.append(label)
            idx += 1

    print(f"  [Augmentation] {len(image_paths)} originals "
          f"+ {idx} augmented = {len(aug_paths)} total")
    return aug_paths, aug_labels


def cleanup_augmented_temp():
    """Remove temporary augmented images after feature extraction."""
    aug_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "augmented_temp"
    )
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
        print("  [Augmentation] Cleaned up augmented_temp/")

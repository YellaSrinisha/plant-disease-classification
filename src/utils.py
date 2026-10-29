"""
utils.py
========
Shared utility functions used across the project.

Responsibilities
----------------
1. DATA DISCOVERY   – scan data/plantvillage/ and data/plantdoc/ folder trees,
                      build a DataFrame with file_path, label, source columns.

2. DATASET SPLIT    – stratified 70 / 15 / 15 train / val / test split,
                      saved as CSV files in data/splits/.

3. CLASS MAPPING    – encode string labels to integers and back
                      (needed for some sklearn internals).

4. FEATURE CACHING  – save / load pre-computed feature arrays as .npz files
                      so feature extraction does not need to re-run every time.

5. LOGGING          – a lightweight console logger with timestamps.

6. REPRODUCIBILITY  – seed setter for numpy and random.

Public API
----------
  discover_dataset(data_dir)                    → pd.DataFrame
  make_splits(df, splits_dir, seed)             → train_df, val_df, test_df
  load_splits(splits_dir)                       → train_df, val_df, test_df
  encode_labels(y)                              → y_int, label_encoder
  save_features(path, X, y)
  load_features(path)                           → X, y
  get_logger(name)                              → logging.Logger
  set_seed(seed)
  print_class_distribution(df, title)
  get_project_root()                            → pathlib.Path
"""

import os
import sys
import random
import logging
import pathlib
import numpy as np
import pandas as pd

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder


# ---------------------------------------------------------------------------
# 1. Data discovery
# ---------------------------------------------------------------------------

# Image file extensions we recognise
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def discover_dataset(data_dir: str) -> pd.DataFrame:
    """
    Walk `data_dir` and collect every image path.

    Expected directory layout (one or two levels of class sub-folders):

        data_dir/
        ├── Tomato_Early_Blight/
        │   ├── img001.jpg
        │   └── img002.jpg
        └── Tomato_Healthy/
            └── img003.jpg

    The source name (e.g. 'plantvillage' or 'plantdoc') is derived from the
    *last component* of data_dir so the function works for both datasets.

    Parameters
    ----------
    data_dir : str – path to a dataset root folder

    Returns
    -------
    pd.DataFrame with columns:
        file_path : str  – absolute path to image file
        label     : str  – class name (sub-folder name)
        source    : str  – dataset name (derived from data_dir basename)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Please place your images under data/plantvillage/ and "
            "data/plantdoc/ with one sub-folder per class."
        )

    source  = os.path.basename(os.path.normpath(data_dir))
    records = []

    for root, dirs, files in os.walk(data_dir):
        # The class label is the immediate parent folder of each image file.
        # Skip the root itself (no label at root level).
        label = os.path.basename(root)
        if label == source:
            continue   # still at the top-level, no class label yet

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                abs_path = os.path.abspath(os.path.join(root, fname))
                records.append({
                    "file_path": abs_path,
                    "label":     label,
                    "source":    source,
                })

    if not records:
        print(f"[WARNING] No images found under: {data_dir}")
        return pd.DataFrame(columns=["file_path", "label", "source"])

    df = pd.DataFrame(records)
    print(f"  [{source}] Found {len(df):,} images across "
          f"{df['label'].nunique()} classes.")
    return df


def build_full_dataframe(
    plantvillage_dir: str,
    plantdoc_dir:     str
) -> pd.DataFrame:
    """
    Discover both datasets and combine into a single DataFrame.

    Parameters
    ----------
    plantvillage_dir : str – path to PlantVillage data root
    plantdoc_dir     : str – path to PlantDoc data root

    Returns
    -------
    pd.DataFrame – combined, shuffled, index-reset DataFrame
    """
    frames = []

    for d in [plantvillage_dir, plantdoc_dir]:
        try:
            frames.append(discover_dataset(d))
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")

    if not frames:
        raise RuntimeError(
            "No dataset directories could be read. "
            "Check that data/plantvillage/ and data/plantdoc/ exist and contain images."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\n  Combined dataset : {len(df):,} images, "
          f"{df['label'].nunique()} classes, "
          f"{df['source'].nunique()} sources")
    return df


# ---------------------------------------------------------------------------
# 2. Stratified dataset split
# ---------------------------------------------------------------------------

def make_splits(
    df:         pd.DataFrame,
    splits_dir: str  = os.path.join("data", "splits"),
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    # test_frac is implied as 1 - train_frac - val_frac = 0.15
    seed:       int   = 42,
    min_class_samples: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a stratified 70 / 15 / 15 train / val / test split and save CSVs.

    Classes with fewer than `min_class_samples` samples are excluded from
    splitting (they remain in train) to prevent stratification errors.

    Parameters
    ----------
    df                : pd.DataFrame – full dataset with 'label' column
    splits_dir        : str          – where to write train/val/test CSV files
    train_frac        : float        – training fraction (default 0.70)
    val_frac          : float        – validation fraction (default 0.15)
    seed              : int          – random state
    min_class_samples : int          – minimum samples per class to stratify

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    os.makedirs(splits_dir, exist_ok=True)

    # Separate classes that are too small to split
    counts      = df["label"].value_counts()
    small_mask  = df["label"].isin(counts[counts < min_class_samples].index)
    df_small    = df[small_mask].copy()
    df_splittable = df[~small_mask].copy()

    if len(df_small) > 0:
        print(f"[INFO] {len(df_small)} samples from {small_mask.sum()} "
              "rare classes will be placed entirely in the training set.")

    # Compute val+test fraction relative to the remaining data after train split
    val_test_frac = 1.0 - train_frac
    val_relative  = val_frac / val_test_frac     # val / (val + test)

    # First split: train vs (val + test)
    train_df, valtest_df = train_test_split(
        df_splittable,
        test_size    = val_test_frac,
        stratify     = df_splittable["label"],
        random_state = seed
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        valtest_df,
        test_size    = 1.0 - val_relative,
        stratify     = valtest_df["label"],
        random_state = seed
    )

    # Add rare-class samples to training
    if len(df_small) > 0:
        train_df = pd.concat([train_df, df_small], ignore_index=True)

    # Shuffle all splits
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = val_df.sample(  frac=1, random_state=seed).reset_index(drop=True)
    test_df  = test_df.sample( frac=1, random_state=seed).reset_index(drop=True)

    # Save CSVs
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(splits_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(splits_dir, "test.csv"),  index=False)

    print(f"\n  Splits saved to: {splits_dir}")
    print(f"    Train : {len(train_df):>6,} samples")
    print(f"    Val   : {len(val_df):>6,} samples")
    print(f"    Test  : {len(test_df):>6,} samples")

    return train_df, val_df, test_df


def load_splits(
    splits_dir: str = os.path.join("data", "splits")
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved train / val / test CSV files from disk.

    Parameters
    ----------
    splits_dir : str – directory containing train.csv, val.csv, test.csv

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    paths = {
        "train": os.path.join(splits_dir, "train.csv"),
        "val":   os.path.join(splits_dir, "val.csv"),
        "test":  os.path.join(splits_dir, "test.csv"),
    }

    dfs = {}
    for split_name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Split file not found: {path}\n"
                "Run make_splits() first to generate the CSV files."
            )
        dfs[split_name] = pd.read_csv(path)
        print(f"  [{split_name:5s}] loaded {len(dfs[split_name]):,} rows from {path}")

    return dfs["train"], dfs["val"], dfs["test"]


# ---------------------------------------------------------------------------
# 3. Label encoding / decoding
# ---------------------------------------------------------------------------

def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to consecutive integers.

    Parameters
    ----------
    y : np.ndarray – 1-D array of string class labels

    Returns
    -------
    y_int          : np.ndarray  – integer-encoded labels
    label_encoder  : LabelEncoder – fitted encoder (use .inverse_transform to decode)
    """
    le    = LabelEncoder()
    y_int = le.fit_transform(y)
    return y_int, le


def decode_labels(y_int: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
    """
    Decode integer labels back to strings.

    Parameters
    ----------
    y_int         : np.ndarray  – integer-encoded labels
    label_encoder : LabelEncoder – the same encoder used in encode_labels()

    Returns
    -------
    np.ndarray – string labels
    """
    return label_encoder.inverse_transform(y_int)


# ---------------------------------------------------------------------------
# 4. Feature caching
# ---------------------------------------------------------------------------

def save_features(path: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Persist a (feature matrix, label vector) pair to a compressed .npz file.

    Parameters
    ----------
    path : str         – output path (e.g. 'data/splits/train_features.npz')
    X    : np.ndarray  – shape (n_samples, n_features)
    y    : np.ndarray  – shape (n_samples,)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    print(f"  [✓] Features cached → {path}  "
          f"(X: {X.shape}, y: {y.shape})")


def load_features(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a (feature matrix, label vector) pair from a .npz file.

    Parameters
    ----------
    path : str – path to the .npz file (with or without the .npz extension)

    Returns
    -------
    X : np.ndarray – feature matrix
    y : np.ndarray – label vector
    """
    # numpy appends .npz automatically; handle both
    npz_path = path if path.endswith(".npz") else path + ".npz"

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Feature cache not found: {npz_path}\n"
            "Run extract_features_from_dataset() first."
        )

    data = np.load(npz_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"  [✓] Features loaded ← {npz_path}  "
          f"(X: {X.shape}, y: {y.shape})")
    return X, y


def features_cached(path: str) -> bool:
    """Return True if a feature cache file already exists at `path`."""
    npz_path = path if path.endswith(".npz") else path + ".npz"
    return os.path.exists(npz_path)


# ---------------------------------------------------------------------------
# 5. Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "plant_disease",
               level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a timestamped console handler.

    Parameters
    ----------
    name  : str – logger name (appears in log lines)
    level : int – logging level (default logging.INFO)

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            fmt   = "%(asctime)s  %(name)s  %(levelname)-8s  %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# 6. Reproducibility seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for numpy and Python's random module to ensure
    reproducible results.

    Parameters
    ----------
    seed : int – seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"  [INFO] Random seed set to {seed}")


# ---------------------------------------------------------------------------
# 7. Diagnostics helpers
# ---------------------------------------------------------------------------

def print_class_distribution(df: pd.DataFrame,
                              title: str = "Class Distribution") -> None:
    """
    Print a sorted table of class counts and percentages.

    Parameters
    ----------
    df    : pd.DataFrame – must contain a 'label' column
    title : str          – heading for the table
    """
    counts   = df["label"].value_counts().sort_index()
    total    = len(df)
    print(f"\n  {title}  (total = {total:,})")
    print(f"  {'Class':<40s}  {'Count':>7}  {'%':>6}")
    print("  " + "-" * 57)
    for cls, cnt in counts.items():
        pct = cnt / total * 100
        print(f"  {cls:<40s}  {cnt:>7,}  {pct:>5.1f}%")
    print()


def print_source_distribution(df: pd.DataFrame) -> None:
    """
    Print how many samples come from each source dataset.

    Parameters
    ----------
    df : pd.DataFrame – must contain 'source' and 'label' columns
    """
    if "source" not in df.columns:
        print("[WARNING] 'source' column not found in DataFrame.")
        return

    print("\n  Source Distribution")
    for src, group in df.groupby("source"):
        print(f"    {src:<15s}: {len(group):>6,} images, "
              f"{group['label'].nunique()} classes")
    print()


def summarise_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame
) -> None:
    """
    Print a quick size / source breakdown of all three splits side by side.
    """
    total = len(train_df) + len(val_df) + len(test_df)
    print("\n  ┌──────────┬──────────┬──────────┬──────────┐")
    print("  │ Split    │  Samples │  Classes │  Sources │")
    print("  ├──────────┼──────────┼──────────┼──────────┤")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_cls = df["label"].nunique()
        n_src = df["source"].nunique() if "source" in df.columns else "–"
        print(f"  │ {name:<8s} │ {len(df):>8,} │ {n_cls:>8} │ {str(n_src):>8} │")
    print("  └──────────┴──────────┴──────────┴──────────┘")
    print(f"  Total images: {total:,}\n")


# ---------------------------------------------------------------------------
# 8. Project root helper
# ---------------------------------------------------------------------------

def get_project_root() -> pathlib.Path:
    """
    Return the absolute path to the project root directory.

    Resolves by walking up from this file until a directory containing
    'requirements.txt' is found.  Falls back to the current working directory.

    Returns
    -------
    pathlib.Path
    """
    here = pathlib.Path(__file__).resolve().parent  # src/
    for candidate in [here.parent, here]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return pathlib.Path.cwd()


def resolve_path(*parts: str) -> str:
    """
    Build an absolute path anchored at the project root.

    Example
    -------
    resolve_path("data", "splits", "train.csv")
    → '/absolute/path/to/project/data/splits/train.csv'
    """
    return str(get_project_root().joinpath(*parts))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Smoke-testing utils.py …\n")

    set_seed(42)

    # Fake DataFrame to test splitting
    rng    = np.random.default_rng(42)
    labels = ["Tomato_Early_Blight", "Tomato_Healthy",
              "Tomato_Late_Blight",  "Pepper_Healthy", "Corn_Gray_Leaf_Spot"]
    n      = 200

    df = pd.DataFrame({
        "file_path": [f"img_{i:04d}.jpg" for i in range(n)],
        "label":     rng.choice(labels, size=n),
        "source":    rng.choice(["plantvillage", "plantdoc"], size=n),
    })

    print_class_distribution(df, "Full Dataset")
    print_source_distribution(df)

    train_df, val_df, test_df = make_splits(
        df, splits_dir=os.path.join("data", "splits_test"), seed=42
    )
    summarise_splits(train_df, val_df, test_df)

    # Test label encoding
    y = train_df["label"].values
    y_int, le = encode_labels(y)
    y_back    = decode_labels(y_int, le)
    assert np.all(y == y_back), "Label encode/decode round-trip failed!"
    print("  Label encode/decode round-trip: OK")

    # Test feature cache
    X_fake = rng.random((len(train_df), 56)).astype(np.float32)
    cache_path = os.path.join("data", "splits_test", "train_features")
    save_features(cache_path, X_fake, y_int)
    X_loaded, y_loaded = load_features(cache_path)
    assert X_loaded.shape == X_fake.shape
    print("  Feature cache save/load: OK")

    print(f"\n  Project root: {get_project_root()}")
    print("\nAll utils smoke-tests passed.")

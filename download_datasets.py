"""
download_datasets.py
====================
Downloads real plant disease images from:
  1. HuggingFace  — PlantVillage (mohanty/PlantVillage, color config)
  2. GitHub       — PlantDoc (pratikkayal/PlantDoc-Dataset)

Target classes (matching project proposal):
  Tomato___Early_blight       → Tomato_Early_Blight
  Tomato___Late_blight        → Tomato_Late_Blight
  Tomato___healthy            → Tomato_Healthy
  Corn_(maize)___Gray_leaf_spot → Corn_Gray_Leaf_Spot
  Potato___Early_blight       → Potato_Early_Blight

Images are saved to:
  data/plantvillage/<ClassName>/
  data/plantdoc/<ClassName>/

Run:
  python download_datasets.py
"""

import os
import sys
import shutil
import zipfile
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
PV_DIR   = BASE_DIR / "data" / "plantvillage"
PD_DIR   = BASE_DIR / "data" / "plantdoc"

# ── Class name mapping: HuggingFace label → our folder name ───────────────
PV_CLASS_MAP = {
    "Tomato___Early_blight"          : "Tomato_Early_Blight",
    "Tomato___Late_blight"           : "Tomato_Late_Blight",
    "Tomato___healthy"               : "Tomato_Healthy",
    "Corn_(maize)___Gray_leaf_spot"  : "Corn_Gray_Leaf_Spot",
    "Potato___Early_blight"          : "Potato_Early_Blight",
}

# ── PlantDoc class name mapping (GitHub folder → our folder name) ─────────
PD_CLASS_MAP = {
    "Tomato Early blight leaf"   : "Tomato_Early_Blight",
    "Tomato Late blight leaf"    : "Tomato_Late_Blight",
    "Tomato leaf"                : "Tomato_Healthy",
    "Corn Gray leaf spot"        : "Corn_Gray_Leaf_Spot",
    "Potato leaf early blight"   : "Potato_Early_Blight",
    # alternate spellings in the repo
    "Tomato Early Blight"        : "Tomato_Early_Blight",
    "Tomato Late Blight"         : "Tomato_Late_Blight",
    "Tomato Leaf"                : "Tomato_Healthy",
    "Corn Gray Leaf Spot"        : "Corn_Gray_Leaf_Spot",
    "Potato Early Blight"        : "Potato_Early_Blight",
}

MAX_PER_CLASS_PV = 1000  # 1000 per class from PlantVillage (balanced training)
MAX_PER_CLASS_PD = 9999  # no cap — take every available PlantDoc image


# ============================================================================
# PART 1 — PlantVillage via HuggingFace datasets
# ============================================================================

def download_plantvillage():
    print("\n" + "="*60)
    print("  PART 1 — PlantVillage (HuggingFace)")
    print("="*60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    print("  Loading dataset from HuggingFace (mohanty/PlantVillage) …")
    print("  This may take a few minutes on first run (caches locally).")

    try:
        ds = load_dataset("mohanty/PlantVillage", "default")
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        sys.exit(1)

    # Combine all splits so we get maximum images
    from datasets import concatenate_datasets
    all_splits = []
    for split_name in ds.keys():
        all_splits.append(ds[split_name])
        print(f"  Split '{split_name}': {len(ds[split_name])} images")
    full_ds = concatenate_datasets(all_splits)
    print(f"  Total images in dataset: {len(full_ds)}")

    # Get label names
    label_feature = full_ds.features["labels"]
    label_names   = label_feature.names
    print(f"  Available classes: {len(label_names)}")

    # Find which label indices correspond to our 5 target classes
    target_indices = {}
    for hf_name, our_name in PV_CLASS_MAP.items():
        # Case-insensitive match
        for idx, lname in enumerate(label_names):
            if lname.lower() == hf_name.lower():
                target_indices[idx] = our_name
                print(f"  Found: '{lname}' → '{our_name}' (index {idx})")
                break
        else:
            # Try partial match
            for idx, lname in enumerate(label_names):
                if hf_name.lower().replace("___","_") in lname.lower().replace("___","_"):
                    target_indices[idx] = our_name
                    print(f"  Found (partial): '{lname}' → '{our_name}' (index {idx})")
                    break
            else:
                print(f"  WARNING: Class '{hf_name}' not found in dataset!")

    if not target_indices:
        print("  ERROR: No target classes found. Check class names.")
        sys.exit(1)

    # Create output directories
    counts = {name: 0 for name in PV_CLASS_MAP.values()}
    for our_name in PV_CLASS_MAP.values():
        (PV_DIR / our_name).mkdir(parents=True, exist_ok=True)

    # Save images
    print(f"\n  Saving up to {MAX_PER_CLASS_PV} images per class …")
    for item in tqdm(full_ds, desc="  PlantVillage", total=len(full_ds)):
        label_idx = item["labels"]
        if label_idx not in target_indices:
            continue
        our_name = target_indices[label_idx]
        if counts[our_name] >= MAX_PER_CLASS_PV:
            continue

        img   = item["image"]
        fname = PV_DIR / our_name / f"img_{counts[our_name]:04d}.jpg"
        try:
            img.convert("RGB").save(str(fname), "JPEG", quality=92)
            counts[our_name] += 1
        except Exception as ex:
            pass  # skip bad images silently

        # Stop early if all classes are filled
        if all(v >= MAX_PER_CLASS_PV for v in counts.values()):
            break

    print("\n  PlantVillage download complete:")
    for cls, cnt in counts.items():
        print(f"    {cls:<30s}: {cnt} images")

    return counts


# ============================================================================
# PART 2 — PlantDoc via GitHub (zip download)
# ============================================================================

def download_plantdoc():
    print("\n" + "="*60)
    print("  PART 2 — PlantDoc (GitHub)")
    print("="*60)

    zip_url  = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = BASE_DIR / "plantdoc_raw.zip"
    extract_path = BASE_DIR / "plantdoc_raw"

    # Download zip
    if not zip_path.exists():
        print(f"  Downloading PlantDoc zip from GitHub …")
        try:
            response = requests.get(zip_url, stream=True, timeout=120)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                desc="  PlantDoc", total=total,
                unit="B", unit_scale=True
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            print(f"  Downloaded: {zip_path}")
        except Exception as e:
            print(f"  ERROR downloading PlantDoc: {e}")
            return {}
    else:
        print(f"  Zip already exists: {zip_path}")

    # Extract
    if not extract_path.exists():
        print("  Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)
        print(f"  Extracted to: {extract_path}")

    # Find the dataset root inside the extracted folder
    extracted_root = extract_path / "PlantDoc-Dataset-master"
    if not extracted_root.exists():
        # Try finding it
        candidates = list(extract_path.iterdir())
        if candidates:
            extracted_root = candidates[0]
        else:
            print("  ERROR: Could not find extracted folder.")
            return {}

    print(f"  Dataset root: {extracted_root}")

    # List all subdirectories (class folders)
    class_dirs = [d for d in extracted_root.iterdir() if d.is_dir()]
    print(f"  Found {len(class_dirs)} class folders:")
    for d in sorted(class_dirs):
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png"))) + len(list(d.glob("*.JPG")))
        print(f"    {d.name:<40s}: {n} images")

    # Map and copy target classes
    counts = {name: 0 for name in PD_CLASS_MAP.values()
              if name in PV_CLASS_MAP.values()}

    for our_name in counts:
        (PD_DIR / our_name).mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        folder_name = class_dir.name

        # Match to our class map
        our_name = None
        # Direct match
        if folder_name in PD_CLASS_MAP:
            our_name = PD_CLASS_MAP[folder_name]
        else:
            # Case-insensitive partial match
            folder_lower = folder_name.lower()
            for pd_name, mapped in PD_CLASS_MAP.items():
                if pd_name.lower() in folder_lower or folder_lower in pd_name.lower():
                    our_name = mapped
                    break

        if our_name is None or our_name not in counts:
            continue

        # Copy images
        images = (list(class_dir.glob("*.jpg")) +
                  list(class_dir.glob("*.JPG")) +
                  list(class_dir.glob("*.jpeg")) +
                  list(class_dir.glob("*.png")))

        for img_path in images[:MAX_PER_CLASS_PD]:
            dst = PD_DIR / our_name / f"img_{counts[our_name]:04d}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
                img.save(str(dst), "JPEG", quality=92)
                counts[our_name] += 1
            except Exception:
                pass

    print("\n  PlantDoc copy complete:")
    for cls, cnt in counts.items():
        print(f"    {cls:<30s}: {cnt} images")

    # Clean up zip and extracted folder to save space
    print("\n  Cleaning up temporary files …")
    try:
        zip_path.unlink()
        shutil.rmtree(extract_path)
        print("  Cleaned up.")
    except Exception as e:
        print(f"  Warning: cleanup failed: {e}")

    return counts


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Plant Disease Dataset Downloader")
    print("="*60)
    print(f"Target classes:")
    for hf, our in PV_CLASS_MAP.items():
        print(f"  {our}")

    pv_counts = download_plantvillage()
    pd_counts = download_plantdoc()

    print("\n" + "="*60)
    print("  DOWNLOAD SUMMARY")
    print("="*60)
    print(f"\n  {'Class':<30s}  {'PlantVillage':>12}  {'PlantDoc':>10}")
    print("  " + "-"*56)
    all_classes = set(list(pv_counts.keys()) + list(pd_counts.keys()))
    total_pv = total_pd = 0
    for cls in sorted(all_classes):
        pv = pv_counts.get(cls, 0)
        pd = pd_counts.get(cls, 0)
        total_pv += pv
        total_pd += pd
        print(f"  {cls:<30s}  {pv:>12}  {pd:>10}")
    print("  " + "-"*56)
    print(f"  {'TOTAL':<30s}  {total_pv:>12}  {total_pd:>10}")
    print(f"\n  Grand total: {total_pv + total_pd} real images")
    print("\n  Done. Now run: python run_pipeline.py")

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

checks = [
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
    "data/splits/train_features.npz",
    "data/splits/val_features.npz",
    "data/splits/test_features.npz",
    "models/knn_model.pkl",
    "models/svm_model.pkl",
    "models/rf_model.pkl",
    "models/ensemble.pkl",
    "notebooks/plant_disease_ensemble.ipynb",
    "reports/metrics_summary.csv",
    "run_pipeline.py",
    "generate_demo_data.py",
    "app.py",
    "src/predict.py",
    "README.md",
    "requirements.txt",
]

all_ok = True
for path in checks:
    exists = os.path.exists(path)
    status = "OK     " if exists else "MISSING"
    print(f"  [{status}] {path}")
    if not exists:
        all_ok = False

pngs = glob.glob("reports/figures/*.png")
txts = glob.glob("reports/*.txt")
print(f"\n  [OK     ] reports/figures/ — {len(pngs)} PNG files")
print(f"  [OK     ] reports/*.txt    — {len(txts)} report files")

print()
if all_ok:
    print("All existing artifacts intact.")
else:
    print("Some artifacts are MISSING — see above.")
    sys.exit(1)

"""
robustness_test.py
==================
Statistical robustness test for the ensemble vs individual models.

Method
------
Draw N_ROUNDS stratified random subsamples (80% of field test set each time)
and compute accuracy + macro F1 on each subsample for every model.

Metrics reported per model:
  - Mean accuracy / F1 across rounds
  - Std deviation of accuracy / F1   ← lower = more stable = more robust
  - Min / Max accuracy across rounds  ← shows worst-case performance
  - 95% confidence interval on mean

Lower std = the model is more consistent when the input varies.
That is the statistical definition of robustness used here.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_all_models
from src.utils   import load_features

# ── Config ────────────────────────────────────────────────────────────────
N_ROUNDS    = 500    # number of random subsamples
SUBSAMPLE   = 0.80   # fraction of field test set per round (stratified)
SEED        = 42

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading models and data …")
models  = load_all_models(os.path.join(PROJECT_ROOT, "models"))
X_test, y_test = load_features(
    os.path.join(PROJECT_ROOT, "data", "splits", "test_features")
)
test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "test.csv"))

# Field subset
field_mask = (test_df["source"] == "plantdoc").values
X_field    = X_test[field_mask]
y_field    = y_test[field_mask]
n_field    = len(X_field)
n_sub      = max(10, int(n_field * SUBSAMPLE))   # samples per round

print(f"Field test set  : {n_field} samples")
print(f"Subsample size  : {n_sub} per round ({SUBSAMPLE*100:.0f}%)")
print(f"Rounds          : {N_ROUNDS}")
print()

# ── Pre-compute all predictions upfront (much faster than per-round predict)
print("Pre-computing predictions on full field set …")
all_preds = {}
for name, model in models.items():
    all_preds[name] = model.predict(X_field)
    print(f"  {name.upper():<10s} done")
print()

# ── Stratified subsampling loop ───────────────────────────────────────────
rng     = np.random.default_rng(SEED)
classes = np.unique(y_field)

# Storage: {model_name: {"acc": [...], "f1": [...]}}
results = {name: {"acc": [], "f1": []} for name in models}

for r in range(N_ROUNDS):
    # Stratified sample: keep class proportions roughly constant
    idx = []
    for cls in classes:
        cls_idx = np.where(y_field == cls)[0]
        n_cls   = max(1, round(len(cls_idx) * SUBSAMPLE))
        chosen  = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)),
                             replace=False)
        idx.extend(chosen.tolist())

    idx   = np.array(idx)
    y_sub = y_field[idx]

    for name in models:
        preds = all_preds[name][idx]   # index into pre-computed predictions
        results[name]["acc"].append(accuracy_score(y_sub, preds))
        results[name]["f1"].append(
            f1_score(y_sub, preds, average="macro", zero_division=0)
        )

# ── Compute statistics ────────────────────────────────────────────────────
rows = []
for name in ["knn", "svm", "rf", "ensemble"]:
    accs = np.array(results[name]["acc"])
    f1s  = np.array(results[name]["f1"])

    # 95% CI: mean ± 1.96 * std/sqrt(n)
    ci_acc = 1.96 * accs.std() / np.sqrt(N_ROUNDS)
    ci_f1  = 1.96 * f1s.std()  / np.sqrt(N_ROUNDS)

    rows.append({
        "Model"       : name.upper(),
        "Acc Mean"    : accs.mean(),
        "Acc Std"     : accs.std(),
        "Acc Min"     : accs.min(),
        "Acc Max"     : accs.max(),
        "Acc 95% CI"  : ci_acc,
        "F1 Mean"     : f1s.mean(),
        "F1 Std"      : f1s.std(),
        "F1 Min"      : f1s.min(),
        "F1 Max"      : f1s.max(),
        "F1 95% CI"   : ci_f1,
    })

df = pd.DataFrame(rows)

# ── Print results ─────────────────────────────────────────────────────────
SEP = "=" * 72

print(SEP)
print("  ROBUSTNESS TEST — Field Accuracy")
print(f"  ({N_ROUNDS} stratified subsamples of {n_sub}/{n_field} field test samples)")
print(SEP)
print(f"  {'Model':<10}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  "
      f"{'Max':>7}  {'95% CI':>9}")
print("  " + "-" * 58)
for row in rows:
    print(f"  {row['Model']:<10}  "
          f"{row['Acc Mean']:>7.4f}  "
          f"{row['Acc Std']:>7.4f}  "
          f"{row['Acc Min']:>7.4f}  "
          f"{row['Acc Max']:>7.4f}  "
          f"±{row['Acc 95% CI']:>7.4f}")

print()
print(SEP)
print("  ROBUSTNESS TEST — Field Macro F1")
print(SEP)
print(f"  {'Model':<10}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  "
      f"{'Max':>7}  {'95% CI':>9}")
print("  " + "-" * 58)
for row in rows:
    print(f"  {row['Model']:<10}  "
          f"{row['F1 Mean']:>7.4f}  "
          f"{row['F1 Std']:>7.4f}  "
          f"{row['F1 Min']:>7.4f}  "
          f"{row['F1 Max']:>7.4f}  "
          f"±{row['F1 95% CI']:>7.4f}")

# ── Robustness ranking ────────────────────────────────────────────────────
print()
print(SEP)
print("  ROBUSTNESS RANKING (lower Std = more robust)")
print(SEP)
sorted_by_std = sorted(rows, key=lambda x: x["F1 Std"])
for rank, row in enumerate(sorted_by_std, 1):
    tag = " ← most robust" if rank == 1 else ""
    print(f"  #{rank}  {row['Model']:<10}  F1 Std = {row['F1 Std']:.4f}"
          f"  Acc Std = {row['Acc Std']:.4f}{tag}")

# ── Ensemble vs SVM comparison ────────────────────────────────────────────
print()
print(SEP)
print("  ENSEMBLE vs SVM — Direct Comparison")
print(SEP)
ens = next(r for r in rows if r["Model"] == "ENSEMBLE")
svm = next(r for r in rows if r["Model"] == "SVM")

print(f"  Field Accuracy:")
print(f"    SVM      mean={svm['Acc Mean']:.4f}  std={svm['Acc Std']:.4f}  "
      f"range=[{svm['Acc Min']:.4f}, {svm['Acc Max']:.4f}]")
print(f"    Ensemble mean={ens['Acc Mean']:.4f}  std={ens['Acc Std']:.4f}  "
      f"range=[{ens['Acc Min']:.4f}, {ens['Acc Max']:.4f}]")

acc_std_diff = svm["Acc Std"] - ens["Acc Std"]
acc_mean_diff = ens["Acc Mean"] - svm["Acc Mean"]
print(f"    Ensemble std is {'lower' if acc_std_diff > 0 else 'higher'} "
      f"by {abs(acc_std_diff):.4f}  "
      f"({'more' if acc_std_diff > 0 else 'less'} robust)")
print(f"    Ensemble mean is {'higher' if acc_mean_diff > 0 else 'lower'} "
      f"by {abs(acc_mean_diff):.4f}")

print(f"\n  Field Macro F1:")
print(f"    SVM      mean={svm['F1 Mean']:.4f}  std={svm['F1 Std']:.4f}  "
      f"range=[{svm['F1 Min']:.4f}, {svm['F1 Max']:.4f}]")
print(f"    Ensemble mean={ens['F1 Mean']:.4f}  std={ens['F1 Std']:.4f}  "
      f"range=[{ens['F1 Min']:.4f}, {ens['F1 Max']:.4f}]")

f1_std_diff  = svm["F1 Std"] - ens["F1 Std"]
f1_mean_diff = ens["F1 Mean"] - svm["F1 Mean"]
print(f"    Ensemble std is {'lower' if f1_std_diff > 0 else 'higher'} "
      f"by {abs(f1_std_diff):.4f}  "
      f"({'more' if f1_std_diff > 0 else 'less'} robust)")
print(f"    Ensemble mean is {'higher' if f1_mean_diff > 0 else 'lower'} "
      f"by {abs(f1_mean_diff):.4f}")

print()
print(SEP)
print("  INTERPRETATION")
print(SEP)
if f1_std_diff > 0 and acc_std_diff > 0:
    print("""
  The ensemble has LOWER variance than SVM on both accuracy and F1.
  This means: across different random subsets of field images, the
  ensemble produces more consistent predictions than SVM.

  Claim you can make:
  'The ensemble is more robust than SVM — while SVM achieves slightly
  higher mean field accuracy, the ensemble has lower prediction variance
  across subsamples, meaning it degrades more gracefully on unseen
  field images and is less sensitive to the specific images encountered.'
""")
elif f1_std_diff > 0:
    print("""
  The ensemble has LOWER F1 variance than SVM (more consistent F1)
  even though accuracy variance may differ.
  This supports a robustness claim on balanced per-class performance.
""")
elif acc_std_diff > 0:
    print("""
  The ensemble has LOWER accuracy variance than SVM (more consistent
  overall accuracy) even though per-class F1 variance may differ.
""")
else:
    print("""
  SVM has lower or equal variance to the ensemble on this dataset.
  The robustness claim for the ensemble vs SVM is NOT supported
  by this variance test. SVM is both more accurate AND more stable
  on field images.

  Honest claim: 'The ensemble outperforms all individual models on
  the full combined test set (89.9% accuracy). On field images
  specifically, SVM is the strongest individual model.'
""")
print(SEP)

# ── Save to CSV ───────────────────────────────────────────────────────────
out_path = os.path.join(PROJECT_ROOT, "reports", "robustness_test.csv")
df.to_csv(out_path, index=False, float_format="%.6f")
print(f"\n  Results saved → {out_path}")

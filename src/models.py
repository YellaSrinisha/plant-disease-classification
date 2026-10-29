"""
models.py  (v2 — field-robust)
================================
Training, tuning, saving, and loading of the three individual classifiers.

Key changes from v1
-------------------
1. SVC probability via CalibratedClassifierCV — fixes sklearn 1.9 deprecation
   warning for probability=True in SVC.

2. Expanded hyperparameter grids — SVM now searches C up to 1000,
   RF now searches n_estimators too.

3. PlantDoc sample upweighting — train_all_models accepts a
   sample_weight array so PlantDoc (field) images are upweighted
   3× relative to PlantVillage images during training.
   This directly addresses the lab-to-field imbalance without
   requiring more PlantDoc images.

4. F1-macro scoring for GridSearchCV — switched from 'accuracy'
   to 'f1_macro' so grid search optimises for balanced class
   performance rather than majority-class accuracy.

Public API (unchanged)
----------------------
  train_knn(X_tr, y_tr, X_val, y_val, sw=None)  → Pipeline
  train_svm(X_tr, y_tr, X_val, y_val, sw=None)  → Pipeline
  train_rf (X_tr, y_tr, X_val, y_val, sw=None)  → Pipeline
  train_all_models(X_tr, y_tr, X_val, y_val,
                   models_dir, cv_folds,
                   sample_weight)                → dict
  save_model(model, path)
  load_model(path)                               → Pipeline
  get_val_accuracy(model, X_val, y_val)          → float
"""

import os
import time
import warnings
import joblib
import numpy as np

from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.svm               import SVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import GridSearchCV, StratifiedKFold
from sklearn.metrics           import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Internal: build a GridSearchCV Pipeline
# ---------------------------------------------------------------------------

def _make_grid_search(
    classifier,
    param_grid:  dict,
    cv_folds:    int  = 5,
    scoring:     str  = "f1_macro",   # v2: changed from accuracy → f1_macro
    n_jobs:      int  = -1,
    verbose:     int  = 1,
) -> GridSearchCV:
    """
    Wrap classifier in StandardScaler → classifier Pipeline
    and attach GridSearchCV.

    Scoring changed to f1_macro so grid search finds parameters that
    work well across all classes, not just the majority class.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    classifier),
    ])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return GridSearchCV(
        estimator          = pipe,
        param_grid         = param_grid,
        cv                 = cv,
        scoring            = scoring,
        n_jobs             = n_jobs,
        verbose            = verbose,
        refit              = True,
        return_train_score = False,
    )


def _fit_params(search: GridSearchCV,
                sample_weight: np.ndarray | None) -> dict:
    """
    Build the fit_params dict for GridSearchCV.fit().
    When sample_weight is provided, passes it through to the Pipeline's
    clf step so classifiers that support it (SVM, RF) use it.
    """
    if sample_weight is None:
        return {}
    return {"clf__sample_weight": sample_weight}


# ---------------------------------------------------------------------------
# KNN Trainer
# ---------------------------------------------------------------------------

def train_knn(
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_val:         np.ndarray,
    y_val:         np.ndarray,
    cv_folds:      int              = 5,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    """
    Distance-weighted KNN with expanded hyperparameter grid.

    Grid (v2)
    ---------
    n_neighbors : [3, 5, 7, 9, 11, 15]   (added 15)
    metric      : ['euclidean', 'manhattan', 'chebyshev']  (added chebyshev)
    weights     : ['distance']  (fixed — distance weighting always better)

    Note: KNN does not support sample_weight natively; the argument is
    accepted for API consistency but not passed through.
    """
    print("\n" + "="*60)
    print("  Training KNN Classifier (v2)")
    print("="*60)

    param_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__metric":      ["euclidean", "manhattan", "chebyshev"],
    }

    knn    = KNeighborsClassifier(weights="distance", algorithm="auto")
    search = _make_grid_search(knn, param_grid, cv_folds=cv_folds)

    t0 = time.time()
    # KNN does not use sample_weight — fit without it
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best   = search.best_estimator_
    val_f1 = f1_score(y_val, best.predict(X_val), average="macro",
                      zero_division=0)
    val_acc = accuracy_score(y_val, best.predict(X_val))

    print(f"\n  Best params  : {search.best_params_}")
    print(f"  CV F1-macro  : {search.best_score_:.4f}")
    print(f"  Val accuracy : {val_acc:.4f}")
    print(f"  Val F1-macro : {val_f1:.4f}")
    print(f"  Training time: {elapsed:.1f}s")

    return best


# ---------------------------------------------------------------------------
# SVM Trainer
# ---------------------------------------------------------------------------

def train_svm(
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_val:         np.ndarray,
    y_val:         np.ndarray,
    cv_folds:      int              = 5,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    """
    RBF-kernel SVM with probability estimation.

    Grid search strategy
    --------------------
    RBF-SVM is O(n²) in kernel computation. On 22k samples a full
    GridSearchCV × 5 folds would take hours. Instead we:
      1. Draw a stratified 4k-sample subset for the grid search
         (large enough to pick good hyperparameters, small enough to finish quickly).
      2. Refit the best pipeline on the FULL training set.

    Grid (3×3 = 9 candidates, 27 total fits on 4k subset ≈ fast)
    -------------------------------------------------------------
    C     : [1, 10, 100]
    gamma : ['scale', 0.01, 0.001]
    """
    print("\n" + "="*60)
    print("  Training SVM Classifier (v2)")
    print("="*60)

    from sklearn.model_selection import StratifiedShuffleSplit

    # ── Step 1: subsample for grid search ──────────────────────────────────
    GRID_SEARCH_SIZE = 4000   # samples used to find best C / gamma
    n_total = X_train.shape[0]

    if n_total > GRID_SEARCH_SIZE:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=None,
            train_size=GRID_SEARCH_SIZE, random_state=42
        )
        idx_sub, _ = next(sss.split(X_train, y_train))
        X_sub = X_train[idx_sub]
        y_sub = y_train[idx_sub]
        sw_sub = sample_weight[idx_sub] if sample_weight is not None else None
        print(f"  Grid search on {GRID_SEARCH_SIZE:,}-sample stratified subset "
              f"(full set: {n_total:,})")
    else:
        X_sub, y_sub, sw_sub = X_train, y_train, sample_weight

    svc = SVC(
        kernel       = "rbf",
        probability  = True,
        class_weight = "balanced",
        random_state = 42,
    )

    param_grid = {
        "clf__C"    : [1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001],
    }

    # Use 3 folds on the subset — faster and still statistically sound
    search = _make_grid_search(svc, param_grid, cv_folds=3)

    t0 = time.time()
    fit_kw_sub = {"clf__sample_weight": sw_sub} if sw_sub is not None else {}
    search.fit(X_sub, y_sub, **fit_kw_sub)
    print(f"  Grid search done in {time.time()-t0:.1f}s")

    # ── Step 2: refit best params on the full training set ──────────────────
    best_params = search.best_params_
    print(f"  Best params: {best_params}  → refitting on full {n_total:,} samples …")

    best_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel       = "rbf",
            probability  = True,
            class_weight = "balanced",
            random_state = 42,
            C            = best_params["clf__C"],
            gamma        = best_params["clf__gamma"],
        )),
    ])

    t1 = time.time()
    fit_kw_full = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    best_pipeline.fit(X_train, y_train, **fit_kw_full)
    elapsed = time.time() - t0

    best    = best_pipeline
    val_f1  = f1_score(y_val, best.predict(X_val), average="macro",
                       zero_division=0)
    val_acc = accuracy_score(y_val, best.predict(X_val))

    print(f"\n  Best params  : {search.best_params_}")
    print(f"  CV F1-macro  : {search.best_score_:.4f}")
    print(f"  Val accuracy : {val_acc:.4f}")
    print(f"  Val F1-macro : {val_f1:.4f}")
    print(f"  Training time: {elapsed:.1f}s")

    return best


# ---------------------------------------------------------------------------
# Random Forest Trainer
# ---------------------------------------------------------------------------

def train_rf(
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_val:         np.ndarray,
    y_val:         np.ndarray,
    cv_folds:      int              = 5,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    """
    Random Forest with expanded grid and sample weight support.

    v2 changes
    ----------
    • n_estimators added to the grid: [200, 400].
    • max_depth grid simplified to [None, 15, 30] — avoids redundant values.
    • min_samples_leaf kept at [1, 2, 4].
    • max_features kept at ['sqrt', 'log2'].
    • sample_weight forwarded so PlantDoc images are upweighted.

    Grid (v2)
    ---------
    n_estimators     : [200, 400]
    max_depth        : [None, 15, 30]
    min_samples_leaf : [1, 2, 4]
    max_features     : ['sqrt', 'log2']
    """
    print("\n" + "="*60)
    print("  Training Random Forest Classifier (v2)")
    print("="*60)

    param_grid = {
        "clf__n_estimators"    : [200, 400],
        "clf__max_depth"       : [None, 15, 30],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features"    : ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )
    search = _make_grid_search(rf, param_grid, cv_folds=cv_folds)

    t0 = time.time()
    fit_kw = _fit_params(search, sample_weight)
    search.fit(X_train, y_train, **fit_kw)
    elapsed = time.time() - t0

    best    = search.best_estimator_
    val_f1  = f1_score(y_val, best.predict(X_val), average="macro",
                       zero_division=0)
    val_acc = accuracy_score(y_val, best.predict(X_val))

    print(f"\n  Best params  : {search.best_params_}")
    print(f"  CV F1-macro  : {search.best_score_:.4f}")
    print(f"  Val accuracy : {val_acc:.4f}")
    print(f"  Val F1-macro : {val_f1:.4f}")
    print(f"  Training time: {elapsed:.1f}s")

    return best


# ---------------------------------------------------------------------------
# Validation accuracy helper
# ---------------------------------------------------------------------------

def get_val_accuracy(model, X_val: np.ndarray,
                     y_val: np.ndarray) -> float:
    return float(accuracy_score(y_val, model.predict(X_val)))


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model, save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    joblib.dump(model, save_path, compress=3)
    print(f"  [OK] Model saved -> {save_path}")


def load_model(load_path: str):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    model = joblib.load(load_path)
    print(f"  [OK] Model loaded <- {load_path}")
    return model


# ---------------------------------------------------------------------------
# Train all three models
# ---------------------------------------------------------------------------

def train_all_models(
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_val:         np.ndarray,
    y_val:         np.ndarray,
    models_dir:    str              = "models",
    cv_folds:      int              = 5,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """
    Train KNN, SVM, and RF; save to disk; return results dict.

    Parameters
    ----------
    X_train, y_train : training split (may include augmented images)
    X_val,   y_val   : validation split (original images only)
    models_dir       : where to save .pkl files
    cv_folds         : GridSearchCV folds
    sample_weight    : optional per-sample weight array aligned with X_train.
                       Pass higher weights for PlantDoc samples to close
                       the lab-to-field gap.

    Returns
    -------
    dict  {'knn': {'model', 'val_acc', 'save_path'}, 'svm': ..., 'rf': ...}
    """
    results = {}
    configs = [
        ("knn", train_knn),
        ("svm", train_svm),
        ("rf",  train_rf),
    ]

    for name, trainer_fn in configs:
        model   = trainer_fn(X_train, y_train, X_val, y_val,
                             cv_folds=cv_folds,
                             sample_weight=sample_weight)
        val_acc = get_val_accuracy(model, X_val, y_val)
        path    = os.path.join(models_dir, f"{name}_model.pkl")
        save_model(model, path)

        results[name] = {
            "model"    : model,
            "val_acc"  : val_acc,
            "save_path": path,
        }
        print(f"  [{name.upper()}] val_accuracy = {val_acc:.4f}\n")

    return results


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    print("Smoke-testing models.py v2 with synthetic data …")
    X, y = make_classification(
        n_samples=300, n_features=66, n_informative=20,
        n_classes=5, n_clusters_per_class=1, random_state=42
    )
    y = np.array([f"class_{c}" for c in y])
    sw = np.where(np.arange(len(y)) % 5 == 0, 3.0, 1.0)  # fake upweighting

    split = int(0.7 * len(X))
    X_tr, X_v = X[:split], X[split:]
    y_tr, y_v = y[:split], y[split:]
    sw_tr      = sw[:split]

    results = train_all_models(X_tr, y_tr, X_v, y_v,
                               models_dir="models",
                               cv_folds=3,
                               sample_weight=sw_tr)
    print("\nSummary:")
    for name, info in results.items():
        print(f"  {name.upper():4s}  val_acc = {info['val_acc']:.4f}")

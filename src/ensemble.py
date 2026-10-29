"""
ensemble.py
===========
Weighted soft-voting ensemble that combines KNN, SVM, and Random Forest.

How it works
------------
Each base model produces a probability distribution over classes via
predict_proba(X).  The ensemble computes a weighted average of those
distributions, where each model's weight equals its validation accuracy
(i.e. better-performing models have more say in the final prediction).

  P_ensemble(class | x) = Σ_i  w_i · P_i(class | x)  /  Σ_i w_i

The class with the highest ensemble probability is the final prediction.

Design notes
------------
• WeightedEnsembleClassifier follows the sklearn estimator interface
  (fit / predict / predict_proba) so it can be dropped into sklearn
  pipelines and evaluation utilities.

• fit() is intentionally lightweight — the base models are already fully
  trained.  It just stores them, normalises the weights, and records the
  class order from the first model.

• The class handles the case where different base models may have been
  trained on slightly different label sets (rare but possible after
  filtering corrupt images) by aligning probability columns to a shared
  class list.

• joblib is used for serialisation; save() / load() are class methods.

Public API
----------
  WeightedEnsembleClassifier(models, weights)
      .fit(X, y)              – record class list, normalise weights
      .predict_proba(X)       – weighted average probability matrix
      .predict(X)             – argmax of predict_proba
      .score(X, y)            – accuracy
      .save(path)             – serialise to disk
  WeightedEnsembleClassifier.load(path)  – class method, returns instance
"""

import os
import numpy as np
import joblib

from sklearn.base    import BaseEstimator, ClassifierMixin
from sklearn.utils   import check_array
from sklearn.metrics import accuracy_score


class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble with per-model accuracy weights.

    Parameters
    ----------
    models  : list of fitted sklearn Pipelines  [knn, svm, rf]
    weights : list of floats                     [w_knn, w_svm, w_rf]
              Typically the validation accuracy of each model.
              Weights are automatically L1-normalised so they sum to 1.

    Attributes (set after fit)
    --------------------------
    classes_         : np.ndarray  – sorted unique class labels
    weights_norm_    : np.ndarray  – L1-normalised weights
    n_models_        : int         – number of base models
    """

    def __init__(self, models: list, weights: list):
        if len(models) != len(weights):
            raise ValueError(
                f"models and weights must have the same length, "
                f"got {len(models)} models and {len(weights)} weights."
            )
        if len(models) == 0:
            raise ValueError("At least one model is required.")

        self.models  = models
        self.weights = weights

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Prepare the ensemble for prediction.

        Does NOT retrain the base models — they must already be fitted.
        Records the class list and normalises the weights.

        Parameters
        ----------
        X : np.ndarray  – feature matrix (used only to derive classes if y
                          contains all expected labels; otherwise ignored)
        y : np.ndarray  – label vector (used to build self.classes_)

        Returns
        -------
        self
        """
        self.classes_      = np.unique(y)
        self.n_models_     = len(self.models)

        # Normalise weights to sum to 1
        w = np.array(self.weights, dtype=np.float64)
        if w.sum() == 0:
            raise ValueError("All weights are zero — cannot normalise.")
        self.weights_norm_ = w / w.sum()

        print(f"  [Ensemble] {self.n_models_} base models, "
              f"{len(self.classes_)} classes")
        print(f"  [Ensemble] Normalised weights: "
              + "  ".join(f"{v:.4f}" for v in self.weights_norm_))

        return self

    # ------------------------------------------------------------------
    # _align_proba  (internal)
    # ------------------------------------------------------------------

    def _align_proba(self,
                     proba:   np.ndarray,
                     model_classes: np.ndarray) -> np.ndarray:
        """
        Reorder / zero-pad a model's probability matrix so its columns
        match the ensemble's self.classes_ exactly.

        This is important when a base model was trained on a subset of
        classes (e.g. after corrupt-image filtering removed all samples
        of one class from training).

        Parameters
        ----------
        proba         : np.ndarray – shape (n_samples, n_model_classes)
        model_classes : np.ndarray – class labels corresponding to columns

        Returns
        -------
        np.ndarray – shape (n_samples, len(self.classes_))
        """
        n_samples    = proba.shape[0]
        n_classes    = len(self.classes_)
        aligned      = np.zeros((n_samples, n_classes), dtype=np.float64)

        # Build a dict from class label → column index in the model output
        model_col = {cls: i for i, cls in enumerate(model_classes)}

        for ens_col, cls in enumerate(self.classes_):
            if cls in model_col:
                aligned[:, ens_col] = proba[:, model_col[cls]]
            # else: leave as 0 (model has never seen this class)

        return aligned

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the weighted average probability over all base models.

        Parameters
        ----------
        X : np.ndarray – feature matrix, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray – shape (n_samples, n_classes), rows sum to 1
        """
        if not hasattr(self, "classes_"):
            raise RuntimeError("Call fit() before predict_proba().")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        weighted_sum = np.zeros((n_samples, n_classes), dtype=np.float64)

        for model, w in zip(self.models, self.weights_norm_):
            # Each model is a sklearn Pipeline with predict_proba support
            proba         = model.predict_proba(X)
            model_classes = model.classes_          # attribute on the clf step

            # The Pipeline exposes .classes_ from its final estimator
            # (sklearn ≥ 1.0 propagates this automatically)
            aligned = self._align_proba(proba, model_classes)
            weighted_sum += w * aligned

        # Re-normalise rows to handle any floating-point drift
        row_sums = weighted_sum.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)   # avoid /0
        return weighted_sum / row_sums

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the class with the highest ensemble probability.

        Parameters
        ----------
        X : np.ndarray – feature matrix, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray – predicted labels, shape (n_samples,)
        """
        proba  = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray – feature matrix
        y : np.ndarray – true labels

        Returns
        -------
        float – accuracy in [0, 1]
        """
        return float(accuracy_score(y, self.predict(X)))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, save_path: str) -> None:
        """
        Serialise the fitted ensemble to disk with joblib.

        Parameters
        ----------
        save_path : str – output file path (e.g. 'models/ensemble.pkl')
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump(self, save_path, compress=3)
        print(f"  [✓] Ensemble saved → {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "WeightedEnsembleClassifier":
        """
        Load a serialised WeightedEnsembleClassifier from disk.

        Parameters
        ----------
        load_path : str – file path to load from

        Returns
        -------
        WeightedEnsembleClassifier – fitted instance
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Ensemble file not found: {load_path}"
            )
        instance = joblib.load(load_path)
        print(f"  [✓] Ensemble loaded ← {load_path}")
        return instance

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        weights_str = ", ".join(f"{w:.4f}" for w in self.weights)
        return (
            f"WeightedEnsembleClassifier("
            f"n_models={len(self.models)}, "
            f"weights=[{weights_str}])"
        )


# ---------------------------------------------------------------------------
# Convenience builder (used from the notebook)
# ---------------------------------------------------------------------------

def build_ensemble(
    model_results: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: str = os.path.join("models", "ensemble.pkl")
) -> WeightedEnsembleClassifier:
    """
    Build, fit, and save a WeightedEnsembleClassifier from the output of
    src.models.train_all_models().

    Parameters
    ----------
    model_results : dict  – output of train_all_models():
                            {'knn': {'model': ..., 'val_acc': ...}, ...}
    X_train       : np.ndarray – training features (for fit() label recording)
    y_train       : np.ndarray – training labels
    save_path     : str        – where to save the ensemble pkl

    Returns
    -------
    WeightedEnsembleClassifier – fitted and saved ensemble
    """
    models  = []
    weights = []

    # Preserve a consistent order: KNN → SVM → RF
    for name in ["knn", "svm", "rf"]:
        if name not in model_results:
            raise KeyError(f"Expected key '{name}' in model_results dict.")
        models.append(model_results[name]["model"])
        weights.append(model_results[name]["val_acc"])

    print("\n" + "="*60)
    print("  Building Weighted Ensemble")
    print("="*60)
    print(f"  KNN  val_acc = {weights[0]:.4f}  → weight")
    print(f"  SVM  val_acc = {weights[1]:.4f}  → weight")
    print(f"  RF   val_acc = {weights[2]:.4f}  → weight")

    ensemble = WeightedEnsembleClassifier(models=models, weights=weights)
    ensemble.fit(X_train, y_train)
    ensemble.save(save_path)

    return ensemble


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets      import make_classification
    from sklearn.pipeline      import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors     import KNeighborsClassifier
    from sklearn.svm           import SVC
    from sklearn.ensemble      import RandomForestClassifier

    print("Smoke-testing ensemble.py with synthetic data …\n")

    # Generate data
    X, y = make_classification(
        n_samples=400, n_features=56, n_informative=20,
        n_classes=5, n_clusters_per_class=1, random_state=0
    )
    y = np.array([f"class_{c}" for c in y])

    split_tr  = int(0.7  * len(X))
    split_val = int(0.85 * len(X))
    X_tr, y_tr = X[:split_tr],       y[:split_tr]
    X_v,  y_v  = X[split_tr:split_val], y[split_tr:split_val]
    X_te, y_te = X[split_val:],      y[split_val:]

    # Quick-fit base models (no GridSearch for speed)
    def _quick_pipe(clf):
        p = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        p.fit(X_tr, y_tr)
        return p

    knn_m = _quick_pipe(KNeighborsClassifier(n_neighbors=5, weights="distance"))
    svm_m = _quick_pipe(SVC(kernel="rbf", probability=True, random_state=0))
    rf_m  = _quick_pipe(RandomForestClassifier(n_estimators=50, random_state=0))

    from sklearn.metrics import accuracy_score as acc
    model_results = {
        "knn": {"model": knn_m, "val_acc": acc(y_v, knn_m.predict(X_v))},
        "svm": {"model": svm_m, "val_acc": acc(y_v, svm_m.predict(X_v))},
        "rf":  {"model": rf_m,  "val_acc": acc(y_v, rf_m.predict(X_v))},
    }

    ens = build_ensemble(model_results, X_tr, y_tr,
                         save_path=os.path.join("models", "ensemble_test.pkl"))

    print(f"\n  Ensemble test accuracy : {ens.score(X_te, y_te):.4f}")
    print(f"  KNN    test accuracy   : {acc(y_te, knn_m.predict(X_te)):.4f}")
    print(f"  SVM    test accuracy   : {acc(y_te, svm_m.predict(X_te)):.4f}")
    print(f"  RF     test accuracy   : {acc(y_te, rf_m.predict(X_te)):.4f}")
    print(f"\n  {ens}")

    # Test save / reload round-trip
    reloaded = WeightedEnsembleClassifier.load(
        os.path.join("models", "ensemble_test.pkl")
    )
    print(f"  Reloaded accuracy      : {reloaded.score(X_te, y_te):.4f}")
    print("\nAll ensemble tests passed.")

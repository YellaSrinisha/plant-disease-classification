"""
evaluation.py
=============
Evaluation, visualisation, and reporting for the plant disease classifiers.

Evaluates each model on two test subsets:
  • Full test set  – all images regardless of source
  • Field subset   – only PlantDoc images (source == 'plantdoc')
                     This specifically targets the lab-to-field performance gap.

Plots generated (all saved to reports/figures/)
-----------------------------------------------
  1. Confusion matrix          – seaborn heatmap per model
  2. Per-class F1 bar chart    – per model
  3. Comparison bar chart      – KNN vs SVM vs RF vs Ensemble (field accuracy)

Reports saved (reports/)
------------------------
  • classification_report_<model>.txt  – full sklearn classification report
  • metrics_summary.csv                – one row per model with key metrics

Public API
----------
  evaluate_model(model, X, y, model_name, label_list) → dict of metrics
  plot_confusion_matrix(cm, labels, title, save_path)
  plot_per_class_f1(report_dict, title, save_path)
  plot_model_comparison(results_dict, save_path)
  evaluate_all(models_dict, X_test, y_test, df_test,
               X_val, y_val, reports_dir, figures_dir)  → summary DataFrame
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# Directories
# ===========================================================================

REPORTS_DIR = os.path.join("reports")
FIGURES_DIR = os.path.join("reports", "figures")


def _ensure_dirs(*dirs):
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ===========================================================================
# Core evaluation helper
# ===========================================================================

def evaluate_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    label_list: list | None = None,
    reports_dir: str = REPORTS_DIR,
    figures_dir: str = FIGURES_DIR,
    save_plots: bool = True,
    suffix: str = ""
) -> dict:
    """
    Evaluate a single model and (optionally) save classification report + plots.

    Parameters
    ----------
    model       : fitted sklearn Pipeline or WeightedEnsembleClassifier
    X           : np.ndarray  – feature matrix
    y_true      : np.ndarray  – true labels
    model_name  : str         – short name used in file names & plot titles
    label_list  : list | None – ordered class labels for axis ticks;
                                derived from y_true if None
    reports_dir : str         – where to save .txt reports
    figures_dir : str         – where to save plots
    save_plots  : bool        – generate and save plots (default True)
    suffix      : str         – appended to file names (e.g. '_field')

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        per_class_f1 (dict), report_str, confusion_matrix (np.ndarray)
    """
    _ensure_dirs(reports_dir, figures_dir)

    y_pred = model.predict(X)

    if label_list is None:
        label_list = sorted(np.unique(np.concatenate([y_true, y_pred])))

    # --- Scalar metrics -------------------------------------------------------
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="macro",
                            zero_division=0, labels=label_list)
    rec   = recall_score(y_true, y_pred, average="macro",
                         zero_division=0, labels=label_list)
    f1    = f1_score(y_true, y_pred, average="macro",
                     zero_division=0, labels=label_list)

    # --- Per-class F1 ---------------------------------------------------------
    f1_per_class_arr = f1_score(y_true, y_pred, average=None,
                                zero_division=0, labels=label_list)
    per_class_f1 = dict(zip(label_list, f1_per_class_arr.tolist()))

    # --- Classification report text ------------------------------------------
    report_str = classification_report(
        y_true, y_pred, labels=label_list, zero_division=0
    )

    # --- Confusion matrix -----------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=label_list)

    # --- Save text report -----------------------------------------------------
    tag        = f"{model_name}{suffix}"
    report_path = os.path.join(reports_dir, f"classification_report_{tag}.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"Classification Report — {tag}\n")
        fh.write("=" * 60 + "\n")
        fh.write(report_str)
        fh.write(f"\nAccuracy  : {acc:.4f}\n")
        fh.write(f"Precision : {prec:.4f}  (macro)\n")
        fh.write(f"Recall    : {rec:.4f}  (macro)\n")
        fh.write(f"F1-score  : {f1:.4f}  (macro)\n")
    print(f"  [✓] Report saved → {report_path}")

    # --- Plots ----------------------------------------------------------------
    if save_plots:
        cm_path  = os.path.join(figures_dir, f"confusion_matrix_{tag}.png")
        f1_path  = os.path.join(figures_dir, f"per_class_f1_{tag}.png")

        plot_confusion_matrix(cm, label_list,
                              title=f"Confusion Matrix — {tag}",
                              save_path=cm_path)
        plot_per_class_f1(per_class_f1,
                          title=f"Per-Class F1 — {tag}",
                          save_path=f1_path)

    return {
        "accuracy":         acc,
        "precision_macro":  prec,
        "recall_macro":     rec,
        "f1_macro":         f1,
        "per_class_f1":     per_class_f1,
        "report_str":       report_str,
        "confusion_matrix": cm,
    }


# ===========================================================================
# Plot 1 – Confusion matrix
# ===========================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    figsize_per_class: float = 0.7
) -> None:
    """
    Draw a seaborn annotated heatmap of a confusion matrix.

    Parameters
    ----------
    cm         : np.ndarray – square confusion matrix (rows=true, cols=pred)
    labels     : list       – class label strings for axis ticks
    title      : str        – plot title
    save_path  : str | None – if given, figure is saved here (PNG, 150 dpi)
    figsize_per_class : float – inches per class label for dynamic sizing
    """
    n = len(labels)
    fig_size = max(8, n * figsize_per_class)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Normalise to percentages (row-wise) for readability
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # avoid divide-by-zero
    cm_pct = cm_norm / row_sums * 100    # 0–100 %

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        linecolor="grey",
        ax=ax,
        cbar_kws={"label": "Row %"},
        annot_kws={"size": max(6, 11 - n // 5)},
    )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n // 8))
    plt.yticks(rotation=0,             fontsize=max(6, 10 - n // 8))
    plt.tight_layout()

    if save_path:
        _ensure_dirs(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Confusion matrix saved → {save_path}")

    plt.close(fig)


# ===========================================================================
# Plot 2 – Per-class F1 bar chart
# ===========================================================================

def plot_per_class_f1(
    per_class_f1: dict,
    title: str = "Per-Class F1 Score",
    save_path: str | None = None,
    threshold: float = 0.5
) -> None:
    """
    Horizontal bar chart of per-class F1 scores.

    Bars below `threshold` are drawn in salmon/red to highlight weak classes.

    Parameters
    ----------
    per_class_f1 : dict        – {class_name: f1_score, ...}
    title        : str         – plot title
    save_path    : str | None  – PNG save path
    threshold    : float       – bars below this value are highlighted red
    """
    classes = list(per_class_f1.keys())
    scores  = [per_class_f1[c] for c in classes]

    # Sort by ascending F1 so the weakest classes appear at the top
    order   = np.argsort(scores)
    classes = [classes[i] for i in order]
    scores  = [scores[i]  for i in order]

    colors  = ["#d9534f" if s < threshold else "#5bc0de" for s in scores]

    n       = len(classes)
    height  = max(5, n * 0.35)
    fig, ax = plt.subplots(figsize=(10, height))

    bars = ax.barh(classes, scores, color=colors, edgecolor="white", height=0.6)

    # Annotate values
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9
        )

    ax.axvline(threshold, color="#f0ad4e", linestyle="--",
               linewidth=1.2, label=f"Threshold = {threshold}")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        _ensure_dirs(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Per-class F1 chart saved → {save_path}")

    plt.close(fig)


# ===========================================================================
# Plot 3 – Model comparison bar chart (field accuracy)
# ===========================================================================

def plot_model_comparison(
    results: dict,
    metric: str = "accuracy",
    title: str = "Model Comparison — Field (PlantDoc) Accuracy",
    save_path: str | None = None
) -> None:
    """
    Grouped bar chart comparing KNN, SVM, RF, and Ensemble on a given metric,
    split into 'full' and 'field' test subsets.

    Parameters
    ----------
    results  : dict  – structure:
                  {
                    'knn':      {'full': {metric: val}, 'field': {metric: val}},
                    'svm':      {...},
                    'rf':       {...},
                    'ensemble': {...},
                  }
    metric   : str   – which metric key to plot (default 'accuracy')
    title    : str   – plot title
    save_path: str | None – PNG save path
    """
    model_names = list(results.keys())
    full_vals   = [results[m].get("full",  {}).get(metric, 0) for m in model_names]
    field_vals  = [results[m].get("field", {}).get(metric, 0) for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, full_vals,  width,
                   label="Full test set",  color="#5bc0de", edgecolor="white")
    bars2 = ax.bar(x + width / 2, field_vals, width,
                   label="Field (PlantDoc)", color="#d9534f", edgecolor="white")

    # Annotate values on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in model_names], fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        _ensure_dirs(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Model comparison chart saved → {save_path}")

    plt.close(fig)


# ===========================================================================
# Master evaluation function
# ===========================================================================

def evaluate_all(
    models_dict:  dict,
    X_test:       np.ndarray,
    y_test:       np.ndarray,
    df_test:      pd.DataFrame,
    reports_dir:  str = REPORTS_DIR,
    figures_dir:  str = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Evaluate all models on the full test set AND the field-only (PlantDoc) subset.
    Generates all plots and saves all text reports.

    Parameters
    ----------
    models_dict : dict  – {'knn': model, 'svm': model, 'rf': model,
                            'ensemble': model}
    X_test      : np.ndarray  – test feature matrix  (aligned with df_test rows)
    y_test      : np.ndarray  – test labels
    df_test     : pd.DataFrame – test split DataFrame with 'source' column
                                 (values: 'plantvillage' or 'plantdoc')
    reports_dir : str  – directory for text reports
    figures_dir : str  – directory for plots

    Returns
    -------
    pd.DataFrame – summary table:
        columns: model, subset, accuracy, precision_macro,
                 recall_macro, f1_macro
    """
    _ensure_dirs(reports_dir, figures_dir)

    # Determine shared label list across all data
    label_list = sorted(np.unique(y_test).tolist())

    # --- Build field (PlantDoc) mask ----------------------------------------
    if "source" in df_test.columns:
        field_mask = df_test["source"].str.lower() == "plantdoc"
    else:
        # Fallback: warn and use an empty field subset
        print("[WARNING] 'source' column not found in df_test. "
              "Field evaluation will be skipped.")
        field_mask = pd.Series([False] * len(df_test), index=df_test.index)

    field_idx = np.where(field_mask.values)[0]

    X_field = X_test[field_idx]
    y_field = y_test[field_idx]

    has_field = len(y_field) > 0

    # --- Evaluate each model ------------------------------------------------
    all_results = {}   # {model_name: {'full': metrics, 'field': metrics}}
    summary_rows = []

    for model_name, model in models_dict.items():
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {model_name.upper()}")
        print(f"{'─'*60}")

        # Full test set
        print("  ▸ Full test set")
        full_metrics = evaluate_model(
            model, X_test, y_test,
            model_name  = model_name,
            label_list  = label_list,
            reports_dir = reports_dir,
            figures_dir = figures_dir,
            save_plots  = True,
            suffix      = "_full"
        )

        # Field (PlantDoc) subset
        if has_field:
            print("  ▸ Field subset (PlantDoc only)")
            field_metrics = evaluate_model(
                model, X_field, y_field,
                model_name  = model_name,
                label_list  = label_list,
                reports_dir = reports_dir,
                figures_dir = figures_dir,
                save_plots  = True,
                suffix      = "_field"
            )
        else:
            print("  [INFO] No PlantDoc samples in test set — skipping field eval.")
            field_metrics = {k: 0.0 for k in
                             ["accuracy", "precision_macro",
                              "recall_macro", "f1_macro"]}

        all_results[model_name] = {
            "full":  full_metrics,
            "field": field_metrics,
        }

        # Summary rows
        for subset_name, metrics in [("full", full_metrics),
                                     ("field", field_metrics)]:
            summary_rows.append({
                "model":            model_name,
                "subset":           subset_name,
                "accuracy":         round(metrics["accuracy"],        4),
                "precision_macro":  round(metrics["precision_macro"], 4),
                "recall_macro":     round(metrics["recall_macro"],    4),
                "f1_macro":         round(metrics["f1_macro"],        4),
            })

    # --- Comparison plot (field accuracy) -----------------------------------
    comparison_path = os.path.join(figures_dir, "model_comparison_accuracy.png")
    plot_model_comparison(
        all_results,
        metric    = "accuracy",
        title     = "Model Comparison — Full vs Field (PlantDoc) Accuracy",
        save_path = comparison_path
    )

    # Also plot F1 macro comparison
    f1_comparison_path = os.path.join(figures_dir, "model_comparison_f1.png")
    plot_model_comparison(
        all_results,
        metric    = "f1_macro",
        title     = "Model Comparison — Full vs Field Macro F1",
        save_path = f1_comparison_path
    )

    # --- Build and save summary DataFrame -----------------------------------
    summary_df = pd.DataFrame(summary_rows)
    csv_path   = os.path.join(reports_dir, "metrics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  [✓] Metrics summary saved → {csv_path}")

    # Pretty-print summary to console
    print("\n" + "="*70)
    print("  EVALUATION SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))

    return summary_df


# ===========================================================================
# Single-model quick report (helper for the notebook)
# ===========================================================================

def print_metrics(metrics: dict, model_name: str = "") -> None:
    """
    Print a concise metrics summary to stdout.

    Parameters
    ----------
    metrics    : dict  – output of evaluate_model()
    model_name : str   – label for the header line
    """
    header = f"  Results — {model_name}" if model_name else "  Results"
    print(f"\n{header}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision_macro']:.4f}  (macro)")
    print(f"  Recall    : {metrics['recall_macro']:.4f}  (macro)")
    print(f"  F1-score  : {metrics['f1_macro']:.4f}  (macro)")


# ===========================================================================
# Quick smoke-test
# ===========================================================================
if __name__ == "__main__":
    from sklearn.datasets      import make_classification
    from sklearn.pipeline      import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble      import RandomForestClassifier

    print("Smoke-testing evaluation.py with synthetic data …\n")

    # Synthetic data
    X, y = make_classification(
        n_samples=300, n_features=56, n_informative=20,
        n_classes=5, n_clusters_per_class=1, random_state=7
    )
    classes = [f"Tomato_{c}" for c in
               ["Early_Blight", "Late_Blight", "Healthy",
                "Leaf_Mold", "Septoria"]]
    y_str   = np.array([classes[i] for i in y])

    # Fake source column for field-subset demo
    sources = np.array(["plantdoc" if i % 4 == 0 else "plantvillage"
                        for i in range(len(y_str))])
    df_test = pd.DataFrame({"file_path": [f"img_{i}.jpg" for i in range(len(y_str))],
                             "label": y_str, "source": sources})

    split = int(0.7 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y_str[:split], y_str[split:]
    df_te       = df_test.iloc[split:].reset_index(drop=True)

    # Quick model
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=30,
                                                    random_state=0))])
    pipe.fit(X_tr, y_tr)

    summary = evaluate_all(
        models_dict  = {"rf": pipe},
        X_test       = X_te,
        y_test       = y_te,
        df_test      = df_te,
        reports_dir  = os.path.join("reports"),
        figures_dir  = os.path.join("reports", "figures"),
    )

    print("\nSmoke-test passed. Check reports/ and reports/figures/ for output.")

"""
app.py
======
Flask web application for Plant Disease Detection.

Routes
------
GET  /          – serves the main upload page
POST /predict   – accepts an image, returns JSON prediction
GET  /health    – health-check endpoint
GET  /about     – project information page

The ML pipeline is loaded once at startup from the parent project's
models/ folder and reused for every prediction (no reloading per request).

Run locally:
    cd plant_disease_webapp
    python app.py

Production (gunicorn):
    gunicorn -w 2 -b 0.0.0.0:5000 app:app
"""

import os
import sys
import time
import logging
import uuid
from pathlib import Path
from collections import defaultdict

from flask import (Flask, request, jsonify, render_template,
                   send_from_directory)
from werkzeug.utils import secure_filename

# ── Path setup ────────────────────────────────────────────────────────────
# Plant_disease_webapp/ is a subdirectory of the main project.
# We add the PARENT directory to sys.path so we can import src.predict etc.
WEBAPP_DIR  = Path(__file__).parent.resolve()
PROJECT_DIR = WEBAPP_DIR.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

# Now import existing ML pipeline (no code duplication)
from src.predict import load_all_models, predict_image
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────
MODELS_DIR      = os.environ.get("MODEL_PATH",  str(PROJECT_DIR / "models"))
UPLOAD_FOLDER   = WEBAPP_DIR / "static" / "uploads"
ALLOWED_EXT     = {"jpg", "jpeg", "png"}
MAX_CONTENT_MB  = 16
RATE_LIMIT_MAX  = 10          # max predictions per minute per IP
RATE_LIMIT_WIN  = 60          # window in seconds

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plant_disease_app")

# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# ── Disease metadata (for richer UI responses) ────────────────────────────
DISEASE_INFO = {
    "Tomato_Early_Blight": {
        "display"     : "Tomato Early Blight",
        "description" : "Caused by Alternaria solani. Brown spots with concentric rings on older leaves.",
        "severity"    : "Moderate",
        "action"      : "Remove infected leaves. Apply copper-based fungicide.",
        "emoji"       : "🍅",
    },
    "Tomato_Late_Blight": {
        "display"     : "Tomato Late Blight",
        "description" : "Caused by Phytophthora infestans. Water-soaked lesions that turn brown.",
        "severity"    : "High",
        "action"      : "Act immediately. Apply systemic fungicide. Remove heavily infected plants.",
        "emoji"       : "🍅",
    },
    "Tomato_Healthy": {
        "display"     : "Tomato — Healthy",
        "description" : "No disease detected. The leaf appears healthy.",
        "severity"    : "None",
        "action"      : "Continue regular care and monitoring.",
        "emoji"       : "✅",
    },
    "Corn_Gray_Leaf_Spot": {
        "display"     : "Corn Gray Leaf Spot",
        "description" : "Caused by Cercospora zeae-maydis. Rectangular gray lesions on leaves.",
        "severity"    : "Moderate",
        "action"      : "Use resistant varieties. Apply foliar fungicide at early stages.",
        "emoji"       : "🌽",
    },
    "Potato_Early_Blight": {
        "display"     : "Potato Early Blight",
        "description" : "Caused by Alternaria solani. Dark brown spots with yellow halos.",
        "severity"    : "Moderate",
        "action"      : "Remove affected foliage. Apply mancozeb or chlorothalonil fungicide.",
        "emoji"       : "🥔",
    },
}

# ── Rate limiting (in-memory, per IP) ─────────────────────────────────────
_rate_store: dict = defaultdict(list)   # ip → list of timestamps

def _is_rate_limited(ip: str) -> bool:
    now  = time.time()
    hits = _rate_store[ip]
    # Keep only timestamps within the window
    hits[:] = [t for t in hits if now - t < RATE_LIMIT_WIN]
    if len(hits) >= RATE_LIMIT_MAX:
        return True
    hits.append(now)
    return False

# ── Model loading (once at startup) ───────────────────────────────────────
_models = None

def get_models():
    """Load models once and cache. Thread-safe for single-worker use."""
    global _models
    if _models is None:
        logger.info("Loading ML models from %s …", MODELS_DIR)
        try:
            _models = load_all_models(MODELS_DIR)
            logger.info("Models loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load models: %s", exc)
            _models = None
    return _models

# ── Helpers ───────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _add_cors(response):
    """Add CORS headers to support mobile app integration."""
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def after_request(response):
    return _add_cors(response)


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/about", methods=["GET"])
def about():
    """Serve the about / project info page."""
    return render_template("about.html")


@app.route("/results", methods=["GET"])
def results():
    """
    Model evaluation results page.
    Reads actual metrics from reports/metrics_summary.csv and
    classification report text files. Never hard-codes any number.
    """
    import csv, os

    reports_dir = PROJECT_DIR / "reports"
    figures_dir = reports_dir / "figures"

    # ── Load metrics_summary.csv ──────────────────────────────────────────
    metrics = []
    csv_path = reports_dir / "metrics_summary.csv"
    csv_available = csv_path.exists()
    if csv_available:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append({
                    "model"           : row["model"].upper(),
                    "subset"          : row["subset"],
                    "accuracy"        : f"{float(row['accuracy'])*100:.1f}%",
                    "precision_macro" : f"{float(row['precision_macro'])*100:.1f}%",
                    "recall_macro"    : f"{float(row['recall_macro'])*100:.1f}%",
                    "f1_macro"        : f"{float(row['f1_macro'])*100:.1f}%",
                })

    # Separate full vs field
    full_metrics  = [m for m in metrics if m["subset"] == "full"]
    field_metrics = [m for m in metrics if m["subset"] == "field"]

    # ── Build trade-off data for "Why Ensemble?" card ─────────────────────
    tradeoff = []
    if csv_available:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader   = csv.DictReader(f)
            rows_raw = list(reader)

        full_raw     = {r["model"].upper(): float(r["accuracy"])
                        for r in rows_raw if r["subset"] == "full"}
        field_raw    = {r["model"].upper(): float(r["accuracy"])
                        for r in rows_raw if r["subset"] == "field"}
        f1_full_raw  = {r["model"].upper(): float(r["f1_macro"])
                        for r in rows_raw if r["subset"] == "full"}
        f1_field_raw = {r["model"].upper(): float(r["f1_macro"])
                        for r in rows_raw if r["subset"] == "field"}

        for model in ["KNN", "SVM", "RF", "ENSEMBLE"]:
            if model in full_raw and model in field_raw:
                gap = full_raw[model] - field_raw[model]
                tradeoff.append({
                    "model"     : model,
                    "full_acc"  : round(full_raw[model]      * 100, 1),
                    "field_acc" : round(field_raw[model]     * 100, 1),
                    "full_f1"   : round(f1_full_raw[model]   * 100, 1),
                    "field_f1"  : round(f1_field_raw[model]  * 100, 1),
                    "gap"       : round(gap                  * 100, 1),
                })

    # ── Load robustness test results ──────────────────────────────────────
    robustness = []
    rob_path   = PROJECT_DIR / "reports" / "robustness_test.csv"
    if rob_path.exists():
        with open(rob_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                robustness.append({
                    "model"   : row["Model"],
                    "f1_mean" : round(float(row["F1 Mean"]) * 100, 1),
                    "f1_std"  : round(float(row["F1 Std"])  * 100, 2),
                    "f1_min"  : round(float(row["F1 Min"])  * 100, 1),
                    "f1_max"  : round(float(row["F1 Max"])  * 100, 1),
                    "acc_std" : round(float(row["Acc Std"]) * 100, 2),
                })

    # ── Load classification report text ───────────────────────────────────
    reports = {}
    for model in ["knn", "svm", "rf", "ensemble"]:
        for subset in ["full", "field"]:
            key  = f"{model}_{subset}"
            path = reports_dir / f"classification_report_{key}.txt"
            if path.exists():
                with open(path, encoding="utf-8", errors="replace") as f:
                    reports[key] = f.read()

    # ── Collect available figure filenames ────────────────────────────────
    figures = {}
    figure_map = {
        "comparison_accuracy" : "model_comparison_accuracy.png",
        "comparison_f1"       : "model_comparison_f1.png",
    }
    for model in ["knn", "svm", "rf", "ensemble"]:
        for subset in ["full", "field"]:
            figure_map[f"cm_{model}_{subset}"]  = f"confusion_matrix_{model}_{subset}.png"
            figure_map[f"f1_{model}_{subset}"]  = f"per_class_f1_{model}_{subset}.png"

    for key, fname in figure_map.items():
        if (figures_dir / fname).exists():
            figures[key] = fname

    return render_template(
        "results.html",
        csv_available  = csv_available,
        full_metrics   = full_metrics,
        field_metrics  = field_metrics,
        reports        = reports,
        figures        = figures,
        tradeoff       = tradeoff,
        robustness     = robustness,
    )


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns 200 if models are loaded, 503 otherwise.
    Used by deployment platforms (Render, Railway) to verify the app is alive.
    """
    models = get_models()
    if models:
        return jsonify({"status": "ok", "models_loaded": True}), 200
    return jsonify({"status": "degraded", "models_loaded": False}), 503


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Accept an uploaded image and return a disease prediction.

    Request
    -------
    multipart/form-data with field 'file' containing the image.

    Response (JSON)
    ---------------
    Success:
    {
      "success"    : true,
      "disease"    : "Tomato_Early_Blight",
      "display"    : "Tomato Early Blight",
      "confidence" : 87.3,
      "description": "...",
      "severity"   : "Moderate",
      "action"     : "...",
      "emoji"      : "🍅",
      "all_probs"  : {"Tomato_Early_Blight": 0.873, ...}
    }

    Error:
    {
      "success": false,
      "error"  : "Human-readable error message"
    }
    """
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({}), 200

    # ── Rate limiting ──────────────────────────────────────────────────────
    ip = request.remote_addr or "unknown"
    if _is_rate_limited(ip):
        logger.warning("Rate limit hit for IP: %s", ip)
        return jsonify({
            "success": False,
            "error"  : "Too many requests. Please wait a moment and try again."
        }), 429

    # ── Model check ────────────────────────────────────────────────────────
    models = get_models()
    if not models:
        return jsonify({
            "success": False,
            "error"  : "Models are not loaded. Please contact support."
        }), 503

    # ── File validation ────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not _allowed(file.filename):
        return jsonify({
            "success": False,
            "error"  : "Unsupported file type. Please upload JPG or PNG."
        }), 400

    # ── Save temporarily ───────────────────────────────────────────────────
    safe_name  = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path   = UPLOAD_FOLDER / unique_name

    try:
        file.save(str(save_path))
    except Exception as exc:
        logger.error("File save failed: %s", exc)
        return jsonify({"success": False, "error": "Failed to save uploaded file."}), 500

    # ── Predict ────────────────────────────────────────────────────────────
    try:
        pil_img = Image.open(str(save_path)).convert("RGB")
        result  = predict_image(pil_img, models)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        _cleanup(save_path)
        return jsonify({
            "success": False,
            "error"  : "Could not process the image. Please try a different photo."
        }), 500
    finally:
        _cleanup(save_path)

    if not result["success"]:
        return jsonify({
            "success": False,
            "error"  : result.get("error", "Prediction failed.")
        }), 500

    # ── Build response ─────────────────────────────────────────────────────
    ens_pred   = result["predictions"]["ensemble"]
    disease    = ens_pred["label"]
    proba_dict = ens_pred.get("proba") or {}
    confidence = round(float(proba_dict.get(disease, 0)) * 100, 1)

    # Get disease metadata (fallback if unknown class)
    meta = DISEASE_INFO.get(disease, {
        "display"    : disease.replace("_", " "),
        "description": "No additional information available.",
        "severity"   : "Unknown",
        "action"     : "Consult an agricultural expert.",
        "emoji"      : "🌿",
    })

    # Format all_probs as percentages for the UI
    all_probs_pct = {
        k.replace("_", " "): round(v * 100, 1)
        for k, v in sorted(proba_dict.items(),
                            key=lambda x: x[1], reverse=True)
    }

    logger.info(
        "Prediction | IP: %s | disease: %s | confidence: %.1f%%",
        ip, disease, confidence
    )

    return jsonify({
        "success"    : True,
        "disease"    : disease,
        "display"    : meta["display"],
        "confidence" : confidence,
        "description": meta["description"],
        "severity"   : meta["severity"],
        "action"     : meta["action"],
        "emoji"      : meta["emoji"],
        "all_probs"  : all_probs_pct,
    })


def _cleanup(path: Path):
    """Delete the temporary uploaded file."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


# ── Serve robots.txt and favicon ──────────────────────────────────────────

@app.route("/robots.txt")
def robots():
    return send_from_directory(str(WEBAPP_DIR), "robots.txt")


@app.route("/report-figure/<filename>")
def report_figure(filename):
    """Serve evaluation plot images from the parent project's reports/figures/."""
    figures_dir = str(PROJECT_DIR / "reports" / "figures")
    # Basic security: only allow PNG files
    if not filename.endswith(".png") or ".." in filename:
        return "", 404
    return send_from_directory(figures_dir, filename)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        str(WEBAPP_DIR / "static"), "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load models at startup so first request is fast
    get_models()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Flask app on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
